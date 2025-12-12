mod protocol;

use wasm_bindgen::prelude::*;
use protocol::{parse_header, parse_payload};

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console)]
    fn log(s: &str);
}

#[wasm_bindgen]
pub fn init_panic_hook() {
    console_error_panic_hook::set_once();
}

#[wasm_bindgen]
pub fn parse_packet(data: &[u8]) -> Result<JsValue, JsValue> {
    let header = parse_header(data)?;
    let packet = parse_payload(data, &header)?;
    Ok(serde_wasm_bindgen::to_value(&packet)?)
}
