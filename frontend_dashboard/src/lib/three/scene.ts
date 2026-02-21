import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { EffectComposer } from 'three/examples/jsm/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/examples/jsm/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/examples/jsm/postprocessing/UnrealBloomPass.js';

class ParticleBufferPool {
    private pool: Float32Array[] = [];
    private poolCapacity: number = 0;
    private readonly maxPoolSize = 10;
    private readonly defaultCapacity = 4096;
    
    acquire(requiredCapacity: number): Float32Array {
        const minCapacity = Math.max(requiredCapacity, this.defaultCapacity);
        for (let i = 0; i < this.pool.length; i++) {
            const buf = this.pool[i];
            if (buf.length >= minCapacity * 3) {
                this.pool.splice(i, 1);
                return buf;
            }
        }
        return new Float32Array(minCapacity * 3);
    }
    
    release(buffer: Float32Array): void {
        if (this.pool.length < this.maxPoolSize) {
            this.pool.push(buffer);
        }
    }
    
    getStats(): { poolSize: number; poolCapacity: number } {
        return { poolSize: this.pool.length, poolCapacity: this.poolCapacity };
    }
}

export class SceneController {
    private scene: THREE.Scene;
    private camera: THREE.PerspectiveCamera;
    private renderer: THREE.WebGLRenderer;
    private controls: OrbitControls;
    private composer: EffectComposer;
    private bloomPass: UnrealBloomPass;
    
    private layers = new Map<number, THREE.Mesh>();
    private particles = new Map<number, THREE.Points>(); // Sparse activations
    private bufferPool = new ParticleBufferPool();
    
    // Theme colors
    private currentTheme = 'magma';
    private themeColors: any = {
        magma: { particle: 0xff4500, block: 0x444444 }, // Orange Red
        matrix: { particle: 0x00ff00, block: 0x003300 }, // Green
        cyberpunk: { particle: 0x00ffff, block: 0x220033 } // Cyan/Purple
    };

    constructor(canvas: HTMLCanvasElement) {
        this.scene = new THREE.Scene();
        this.camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
        this.renderer = new THREE.WebGLRenderer({ canvas, antialias: false });
        this.renderer.setSize(window.innerWidth, window.innerHeight);
        this.renderer.setPixelRatio(window.devicePixelRatio);

        this.controls = new OrbitControls(this.camera, canvas);
        this.camera.position.set(0, 5, 20);
        this.controls.update();

        // Lighting
        const ambientLight = new THREE.AmbientLight(0x404040);
        this.scene.add(ambientLight);
        const directionalLight = new THREE.DirectionalLight(0xffffff, 0.5);
        directionalLight.position.set(1, 1, 1);
        this.scene.add(directionalLight);
        
        // Post-processing
        const renderScene = new RenderPass(this.scene, this.camera);
        this.bloomPass = new UnrealBloomPass(new THREE.Vector2(window.innerWidth, window.innerHeight), 1.5, 0.4, 0.85);
        this.bloomPass.enabled = false;
        
        this.composer = new EffectComposer(this.renderer);
        this.composer.addPass(renderScene);
        this.composer.addPass(this.bloomPass);

        this.animate();
    }

    public setBloom(enabled: boolean) {
        this.bloomPass.enabled = enabled;
    }
    
    public setTheme(themeName: string) {
        if (!this.themeColors[themeName]) return;
        this.currentTheme = themeName;
        const colors = this.themeColors[themeName];
        
        this.particles.forEach(p => {
             (p.material as THREE.PointsMaterial).color.setHex(colors.particle);
        });
        this.layers.forEach(l => {
             (l.material as THREE.MeshBasicMaterial).color.setHex(colors.block);
        });
    }

    public resize(width: number, height: number) {
        this.camera.aspect = width / height;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(width, height);
        this.composer.setSize(width, height);
    }

    // Topology State
    private layerTopology = new Map<number, { width: number; height: number; z: number }>();
    private ghostLayers = new Map<number, THREE.LineSegments>();
    private isTopologySetup = false;
    private lastTopologyLayerCount = 0;

    private static fallbackGrid(neuronCount: number): { width: number; height: number } {
        const safeCount = Math.max(1, Math.floor(neuronCount));
        const width = Math.ceil(Math.sqrt(safeCount));
        const height = Math.ceil(safeCount / width);
        return { width, height };
    }

    private clearTopology() {
        for (const layer of this.ghostLayers.values()) {
            this.scene.remove(layer);
            layer.geometry.dispose();
            (layer.material as THREE.Material).dispose();
        }
        this.ghostLayers.clear();
        this.layerTopology.clear();
        this.layers.clear();

        for (const cloud of this.particles.values()) {
            this.scene.remove(cloud);
            const geom = cloud.geometry as THREE.BufferGeometry;
            const posAttr = geom.getAttribute('position') as THREE.BufferAttribute;
            if (posAttr && posAttr.array) {
                this.bufferPool.release(posAttr.array as Float32Array);
            }
            geom.dispose();
            (cloud.material as THREE.Material).dispose();
        }
        this.particles.clear();

        this.isTopologySetup = false;
        this.lastTopologyLayerCount = 0;
    }

    public setupTopology(meta: any) {
        const layers = meta?.layers ?? [];
        if (!Array.isArray(layers) || layers.length === 0) {
            return; // don't lock in an empty topology
        }

        if (this.isTopologySetup && layers.length === this.lastTopologyLayerCount) {
            return;
        }

        if (this.isTopologySetup) {
            this.clearTopology();
        }

        console.log('[Scene] Setting up topology:', meta);
        
        const layerGap = 2.0; // Distance between layers
        
        let maxW = 1;
        let maxH = 1;

        layers.forEach((layer: any, layerIndex: number) => {
            const neuronCount = Number(layer.neuron_count ?? 0);
            const recW = Number(layer.recommended_width ?? 0);
            const recH = Number(layer.recommended_height ?? 0);
            const grid =
                recW > 0 && recH > 0
                    ? { width: recW, height: recH }
                    : SceneController.fallbackGrid(neuronCount);

            maxW = Math.max(maxW, grid.width);
            maxH = Math.max(maxH, grid.height);

            // Store topology
            this.layerTopology.set(layer.layer_id, {
                width: grid.width,
                height: grid.height,
                z: layerIndex * -layerGap // Stack backwards by index (stable)
            });
            
            // Create Ghost Plane (Grid Helper)
            const w = grid.width;
            const h = grid.height;
            const spacing = 0.2;
            
            const geometry = new THREE.PlaneGeometry(w * spacing, h * spacing);
            const edges = new THREE.EdgesGeometry(geometry);
            const line = new THREE.LineSegments(edges, new THREE.LineBasicMaterial({
                color: 0x333333,
                transparent: true,
                opacity: 0.3
            }));
            
            line.position.z = layerIndex * -layerGap;
            this.scene.add(line);
            this.ghostLayers.set(layer.layer_id, line);
            this.layers.set(layer.layer_id, line as any); // for theme updates
        });
        
        // Auto-Frame Camera
        const stackDepth = Math.max(1, layers.length - 1) * layerGap;
        const halfW = (maxW * 0.2) / 2;
        const halfH = (maxH * 0.2) / 2;
        const radius = Math.max(halfW, halfH, stackDepth * 0.5) + 2.0;

        this.controls.target.set(0, 0, -stackDepth / 2);
        this.camera.near = Math.max(0.01, radius / 100);
        this.camera.far = radius * 50;
        this.camera.updateProjectionMatrix();
        this.camera.position.set(radius * 1.2, radius * 0.8, radius * 1.2);
        this.controls.update();
        
        this.isTopologySetup = true;
        this.lastTopologyLayerCount = layers.length;
    }

    public updateLayers(summaries: any[]) {
        if (!summaries || summaries.length === 0) return;

        // If topology exists, color the ghost grids by mean activity for macro readability.
        if (this.isTopologySetup) {
            // If we bootstrapped topology from the first summary (1 layer),
            // rebuild when more layer summaries arrive so the full stack appears.
            if (summaries.length > this.lastTopologyLayerCount) {
                this.setupTopology({ layers: summaries });
            }

            for (const s of summaries) {
                const layerId = Number(s.layer_id);
                const mean = Number(s.mean ?? 0);
                const line = this.ghostLayers.get(layerId);
                if (!line) continue;

                const intensity = Math.max(0, Math.min(1, mean)); // assume normalized-ish
                const base = this.themeColors[this.currentTheme].block;
                const color = new THREE.Color(base).lerp(new THREE.Color(this.themeColors[this.currentTheme].particle), intensity);
                (line.material as THREE.LineBasicMaterial).color.copy(color);
                (line.material as THREE.LineBasicMaterial).opacity = 0.15 + 0.35 * intensity;
            }
            return;
        }

        // Fallback: if no topology meta arrived, build a minimal topology from summaries.
        const layerGap = 2.0;
        const layers = summaries.map((s, i) => ({
            layer_id: Number(s.layer_id),
            neuron_count: Number(s.neuron_count ?? 0),
            recommended_width: 0,
            recommended_height: 0,
            _index: i,
        }));
        this.setupTopology({ layers });
    }

    public updateSparse(sparse: any) {
        const layerId = sparse.layer_id;
        const topology = this.layerTopology.get(layerId);
        if (!topology) return;
        
        let pointCloud = this.particles.get(layerId);
        
        if (!pointCloud) {
            const geometry = new THREE.BufferGeometry();
            const positions = this.bufferPool.acquire(4096);
            
            geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
            geometry.setDrawRange(0, 0);
            
            const material = new THREE.PointsMaterial({ 
                color: this.themeColors[this.currentTheme].particle,
                size: 0.15,
                sizeAttenuation: true,
                transparent: true,
                opacity: 0.9,
                blending: THREE.AdditiveBlending,
                depthWrite: false
            });
            
            pointCloud = new THREE.Points(geometry, material);
            pointCloud.position.z = topology.z;
            this.scene.add(pointCloud);
            this.particles.set(layerId, pointCloud);
        }
        
        const indices: number[] = Array.isArray(sparse.indices) ? sparse.indices : [];
        const values: number[] = Array.isArray(sparse.values) ? sparse.values : [];

        const desiredCount = Math.min(indices.length, values.length);

        const geometry = pointCloud.geometry as THREE.BufferGeometry;
        const positionAttr = geometry.getAttribute('position') as THREE.BufferAttribute;
        let positions = positionAttr.array as Float32Array;
        const currentCapacity = positions.length / 3;

        const count = Math.min(desiredCount, 200_000);
        if (count > currentCapacity) {
            this.bufferPool.release(positions);
            const nextCapacity = Math.min(200_000, Math.ceil(count * 1.25));
            positions = this.bufferPool.acquire(nextCapacity);
            geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
        }
        const width = topology.width;
        const height = topology.height;
        const spacing = 0.2;
        
        const offsetX = (width * spacing) / 2;
        const offsetY = (height * spacing) / 2;

        for (let i = 0; i < count; i++) {
            const idx = indices[i] ?? 0;
            
            const col = idx % width;
            const row = Math.floor(idx / width);
            
            const x = (col * spacing) - offsetX;
            const y = (row * spacing) - offsetY;
            
            positions[i * 3] = x;
            positions[i * 3 + 1] = y;
            positions[i * 3 + 2] = 0;
        }
        
        geometry.setDrawRange(0, count);
        (geometry.getAttribute('position') as THREE.BufferAttribute).needsUpdate = true;
    }

    private animate = () => {
        requestAnimationFrame(this.animate);
        this.controls.update();
        this.composer.render();
    };
}
