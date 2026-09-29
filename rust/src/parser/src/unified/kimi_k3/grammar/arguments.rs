// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Kimi K3 XTML tool-call grammar: call tags and their typed arguments.

use serde_json::{Map, Value};
use xgrammar_structural_tag::format::{Format, JsonSchemaFormat, TagFormat};
use xgrammar_structural_tag::tool::{FunctionToolParam, function_parameters};

use super::super::{ARG_CLOSE, CALL_CLOSE, JSON_CLOSE, JSON_OPEN, OPEN, SEP};

const XTML_TYPES: &[&str] = &["string", "number", "boolean", "null", "object", "array"];

/// Build the tag for one call to `tool`.
pub(super) fn call_tag(tool: &FunctionToolParam) -> TagFormat {
    let parameters = function_parameters(&tool.function);
    let call_body = Format::or(vec![
        typed_arguments(&parameters),
        raw_json_arguments(&parameters),
    ]);

    TagFormat::new(
        format!(
            "{OPEN}call tool=\"{}\" index=\"",
            escape_attr_value(&tool.function.name)
        ),
        Format::sequence(vec![
            Format::regex("[1-9][0-9]*"),
            Format::const_string(format!("\"{SEP}")),
            call_body,
        ]),
        CALL_CLOSE,
    )
}

fn typed_arguments(parameters: &Value) -> Format {
    let Some(schema) = parameters.as_object() else {
        return if parameters == &Value::Bool(false) {
            Format::const_string("")
        } else {
            Format::star(permissive_argument())
        };
    };
    let Some(properties) = schema.get("properties").and_then(Value::as_object) else {
        return Format::star(permissive_argument());
    };
    if properties.is_empty() {
        return Format::star(permissive_argument());
    }

    let root_defs = root_definitions(schema);
    let arguments = properties
        .iter()
        .flat_map(|(key, schema)| argument_tags(key, schema, &root_defs))
        .map(Format::Tag)
        .collect::<Vec<_>>();
    let arguments = match arguments.as_slice() {
        [argument] => argument.clone(),
        _ => Format::or(arguments),
    };
    // Keep typed arguments order-agnostic and non-unique, but do not allow an
    // empty call when the root schema declares required properties.
    if schema
        .get("required")
        .and_then(Value::as_array)
        .is_some_and(|required| !required.is_empty())
    {
        Format::plus(arguments)
    } else {
        Format::star(arguments)
    }
}

fn argument_tags(key: &str, schema: &Value, root_defs: &Map<String, Value>) -> Vec<TagFormat> {
    let types = schema_types(schema);
    types
        .into_iter()
        .map(|xtml_type| {
            let content = if xtml_type == "string" {
                string_argument_content(schema)
            } else {
                json_schema(attach_root_definitions(
                    &narrow_schema_type(schema, xtml_type),
                    root_defs,
                ))
            };
            TagFormat::new(
                format!(
                    "{OPEN}argument key=\"{}\" type=\"{xtml_type}\"{SEP}",
                    escape_attr_value(key)
                ),
                content,
                ARG_CLOSE,
            )
        })
        .collect()
}

fn schema_types(schema: &Value) -> Vec<&'static str> {
    let Some(schema) = schema.as_object() else {
        return XTML_TYPES.to_vec();
    };
    let mut types = Vec::new();
    match schema.get("type") {
        Some(Value::String(value)) => push_schema_type(&mut types, value),
        Some(Value::Array(values)) => {
            for value in values.iter().filter_map(Value::as_str) {
                push_schema_type(&mut types, value);
            }
        }
        _ => {}
    }
    if types.is_empty()
        && let Some(value) = schema.get("const")
    {
        push_value_type(&mut types, value);
    }
    if types.is_empty()
        && let Some(values) = schema.get("enum").and_then(Value::as_array)
    {
        for value in values {
            push_value_type(&mut types, value);
        }
    }
    if types.is_empty() {
        XTML_TYPES.to_vec()
    } else {
        types
    }
}

fn push_value_type(types: &mut Vec<&'static str>, value: &Value) {
    let xtml_type = match value {
        Value::String(_) => "string",
        Value::Number(_) => "number",
        Value::Bool(_) => "boolean",
        Value::Null => "null",
        Value::Object(_) => "object",
        Value::Array(_) => "array",
    };
    if !types.contains(&xtml_type) {
        types.push(xtml_type);
    }
}

fn narrow_schema_type(schema: &Value, xtml_type: &str) -> Value {
    let Some(mut schema) = schema.as_object().cloned() else {
        return schema.clone();
    };
    let json_type = if xtml_type == "number" && explicitly_integer_only(&schema) {
        "integer"
    } else {
        xtml_type
    };
    schema.insert("type".to_string(), Value::String(json_type.to_string()));
    Value::Object(schema)
}

fn explicitly_integer_only(schema: &Map<String, Value>) -> bool {
    match schema.get("type") {
        Some(Value::String(value)) => value == "integer",
        Some(Value::Array(values)) => {
            let values = values.iter().filter_map(Value::as_str).collect::<Vec<_>>();
            values.contains(&"integer") && !values.contains(&"number")
        }
        _ => false,
    }
}

fn push_schema_type(types: &mut Vec<&'static str>, json_type: &str) {
    let xtml_type = match json_type {
        "string" => Some("string"),
        "integer" | "number" => Some("number"),
        "boolean" => Some("boolean"),
        "null" => Some("null"),
        "object" => Some("object"),
        "array" => Some("array"),
        _ => None,
    };
    if let Some(xtml_type) = xtml_type
        && !types.contains(&xtml_type)
    {
        types.push(xtml_type);
    }
}

fn string_argument_content(schema: &Value) -> Format {
    let Some(schema) = schema.as_object() else {
        return Format::any_text_excluding(&[ARG_CLOSE, CALL_CLOSE]);
    };
    let values = schema
        .get("enum")
        .and_then(Value::as_array)
        .cloned()
        .or_else(|| schema.get("const").cloned().map(|value| vec![value]));
    let Some(values) = values else {
        return Format::any_text_excluding(&[ARG_CLOSE, CALL_CLOSE]);
    };
    if values.is_empty()
        || values.len() > 256
        || values
            .iter()
            .any(|value| value.as_str().is_none_or(|value| value.contains("<|")))
    {
        return Format::any_text_excluding(&[ARG_CLOSE, CALL_CLOSE]);
    }
    let values = values.iter().filter_map(Value::as_str).collect::<Vec<_>>();
    match values.as_slice() {
        [value] => Format::const_string(*value),
        _ => Format::or(values.into_iter().map(Format::const_string).collect()),
    }
}

fn raw_json_arguments(parameters: &Value) -> Format {
    Format::tag(
        format!("{JSON_OPEN} type=\"object\"{SEP}"),
        json_schema(parameters.clone()),
        JSON_CLOSE,
    )
}

fn permissive_argument() -> Format {
    let key = Format::regex(r#"(?:[^<\"&]|&(?:amp|quot);|<[^|])*"#);
    let alternatives = XTML_TYPES
        .iter()
        .map(|xtml_type| {
            Format::sequence(vec![
                key.clone(),
                Format::const_string(format!("\" type=\"{xtml_type}\"{SEP}")),
                if *xtml_type == "string" {
                    Format::any_text_excluding(&[ARG_CLOSE, CALL_CLOSE])
                } else {
                    Format::json_schema(Value::Bool(true))
                },
            ])
        })
        .collect();
    Format::tag(
        format!("{OPEN}argument key=\""),
        Format::or(alternatives),
        ARG_CLOSE,
    )
}

fn json_schema(schema: Value) -> Format {
    Format::JsonSchema(JsonSchemaFormat::new(schema))
}

fn root_definitions(schema: &Map<String, Value>) -> Map<String, Value> {
    ["$defs", "definitions"]
        .into_iter()
        .filter_map(|key| schema.get(key).map(|value| (key.to_string(), value.clone())))
        .collect()
}

fn attach_root_definitions(schema: &Value, root_defs: &Map<String, Value>) -> Value {
    let Some(mut schema) = schema.as_object().cloned() else {
        return schema.clone();
    };
    for (key, value) in root_defs {
        schema.entry(key.clone()).or_insert_with(|| value.clone());
    }
    Value::Object(schema)
}

fn escape_attr_value(value: &str) -> String {
    value.replace('&', "&amp;").replace('"', "&quot;")
}

#[cfg(test)]
mod tests {
    use expect_test::expect;
    use serde_json::{Map, json};
    use xgrammar_structural_tag::FunctionDefinition;
    use xgrammar_structural_tag::tool::FunctionToolParam;

    #[test]
    fn call_tag_matches_xtml_arguments() {
        let tool = FunctionToolParam::new(FunctionDefinition::new("get_weather").with_parameters(
            json!({
                "$defs": {
                    "place": { "type": "object", "properties": { "city": { "type": "string" } } }
                },
                "type": "object",
                "properties": {
                    "unit": { "type": "string", "enum": ["celsius", "fahrenheit"] },
                    "place": { "$ref": "#/$defs/place", "type": "object" }
                },
                "required": ["place"]
            }),
        ));

        expect![[r##"{"begin":"<|open|>call tool=\"get_weather\" index=\"","content":{"type":"sequence","elements":[{"type":"regex","pattern":"[1-9][0-9]*"},{"type":"const_string","value":"\"<|sep|>"},{"type":"or","elements":[{"type":"plus","content":{"type":"or","elements":[{"type":"tag","begin":"<|open|>argument key=\"unit\" type=\"string\"<|sep|>","content":{"type":"or","elements":[{"type":"const_string","value":"celsius"},{"type":"const_string","value":"fahrenheit"}]},"end":"<|close|>argument<|sep|>"},{"type":"tag","begin":"<|open|>argument key=\"place\" type=\"object\"<|sep|>","content":{"type":"json_schema","json_schema":{"$ref":"#/$defs/place","type":"object","$defs":{"place":{"type":"object","properties":{"city":{"type":"string"}}}}},"style":"json","any_order":false,"max_whitespace_cnt":null,"excludes":[]},"end":"<|close|>argument<|sep|>"}]}},{"type":"tag","begin":"<|open|>json type=\"object\"<|sep|>","content":{"type":"json_schema","json_schema":{"$defs":{"place":{"type":"object","properties":{"city":{"type":"string"}}}},"type":"object","properties":{"unit":{"type":"string","enum":["celsius","fahrenheit"]},"place":{"$ref":"#/$defs/place","type":"object"}},"required":["place"]},"style":"json","any_order":false,"max_whitespace_cnt":null,"excludes":[]},"end":"<|close|>json<|sep|>"}]}]},"end":"<|close|>call<|sep|>"}"##]].assert_eq(&serde_json::to_string(&super::call_tag(&tool)).unwrap());
    }

    #[test]
    fn typed_arguments_require_one_tag_only_for_nonempty_required() {
        let required = super::typed_arguments(&json!({
            "type": "object",
            "properties": { "query": { "type": "string" } },
            "required": ["query"]
        }));
        let optional = super::typed_arguments(&json!({
            "type": "object",
            "properties": { "query": { "type": "string" } }
        }));
        let empty_required = super::typed_arguments(&json!({
            "type": "object",
            "properties": { "query": { "type": "string" } },
            "required": []
        }));

        assert_eq!(serde_json::to_value(required).unwrap()["type"], "plus");
        assert_eq!(serde_json::to_value(optional).unwrap()["type"], "star");
        assert_eq!(
            serde_json::to_value(empty_required).unwrap()["type"],
            "star"
        );
    }

    #[test]
    fn union_argument_content_matches_its_xtml_type() {
        let tags = super::argument_tags(
            "count",
            &json!({ "type": ["integer", "null"] }),
            &Map::new(),
        );
        let tags = serde_json::to_string(&tags).unwrap();

        assert!(tags.contains(r#"type=\"number\""#));
        assert!(
            tags.contains(r#""json_schema":{"type":"integer"}"#),
            "{tags}"
        );
        assert!(tags.contains(r#"type=\"null\""#));
        assert!(tags.contains(r#""json_schema":{"type":"null"}"#), "{tags}");
    }

    #[test]
    fn unsafe_string_enum_falls_back_as_a_whole() {
        let format = super::string_argument_content(&json!({
            "type": "string",
            "enum": ["safe", "<|unsafe"]
        }));

        assert_eq!(serde_json::to_value(format).unwrap()["type"], "any_text");
    }
}
