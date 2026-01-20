mod protocol;

#[cfg(test)]
mod protocol_tests;

use wasm_bindgen::prelude::*;
use protocol::{parse_header, parse_payload, ParseError};

impl ParseError {
    fn to_js_error(&self) -> JsValue {
        let msg = match self {
            ParseError::BufferTooShort => "Buffer too short for header".to_string(),
            ParseError::InvalidMagic => "Invalid magic number".to_string(),
            ParseError::PayloadTooShort => "Payload too short".to_string(),
            ParseError::PayloadTruncated => "Payload truncated".to_string(),
            ParseError::UnsupportedMessageType(t) => format!("Unsupported message type: {}", t),
            ParseError::InvalidVersion(v) => format!("Invalid version: {}", v),
        };
        JsError::new(&msg).into()
    }
}

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
    let header = parse_header(data).map_err(|e| e.to_js_error())?;
    let packet = parse_payload(data, &header).map_err(|e| e.to_js_error())?;
    Ok(serde_wasm_bindgen::to_value(&packet)?)
}
