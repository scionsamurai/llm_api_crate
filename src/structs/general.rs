// src/structs/general.rs
use serde::{Deserialize, Serialize};
use pyo3::prelude::*;

#[derive(Debug, Serialize, Clone)]
pub struct Message {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Content {
    pub role: String,
    pub parts: Vec<Part>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Part {
    pub text: String,
}

impl<'source> FromPyObject<'source> for Message {
    fn extract(ob: &'source PyAny) -> PyResult<Self> {
        let dict = ob.downcast::<pyo3::types::PyDict>()?;
        
        let role: String = dict.get_item("role")
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>("Missing 'role' key"))?
            .extract()?;
            
        let content: String = dict.get_item("content")
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>("Missing 'content' key"))?
            .extract()?;
            
        Ok(Message { role, content })
    }
}