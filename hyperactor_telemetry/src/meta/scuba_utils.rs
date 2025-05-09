use std::time::Duration;

use fbinit::was_performed;
use opentelemetry::Array;
use opentelemetry::Key;
use opentelemetry::Value;
use opentelemetry_sdk::Resource;
use scuba::ScubaSample;
use scuba::ScubaValue;

pub const TABLE: Key = Key::from_static_str("meta.scuba.table");
pub const SUBSET: Key = Key::from_static_str("meta.scuba.subset");
pub const COLUMNS: Key = Key::from_static_str("meta.scuba.columns");

pub(crate) fn with_value(
    key: &str,
    value: &Value,
    mut sample: scuba::ScubaSample,
) -> scuba::ScubaSample {
    match value {
        Value::String(s) => sample.add(key, s.as_str()),
        Value::Bool(b) => sample.add(key, if *b { 1 } else { 0 }),
        Value::I64(i) => sample.add(key, *i),
        Value::F64(f) => sample.add(key, *f),
        Value::Array(Array::String(array)) => sample.add(
            key,
            ScubaValue::NormVector(array.iter().map(|x| String::from(x.as_str())).collect()),
        ),
        _ => sample.add(key, format!("{}", value)),
    };
    sample
}

pub(crate) fn add_resource_to_sample(r: &Resource, mut sample: ScubaSample) -> ScubaSample {
    if let Some(Value::Array(Array::String(columns))) = r.get(&COLUMNS) {
        for column in columns.iter() {
            if let Some(val) = r.get(&Key::new(column.to_string())) {
                sample = with_value(column.as_str(), &val, sample);
            }
        }
    }
    sample
}

pub(crate) struct ScubaClientWrapper {
    client: scuba::ScubaClient,
    base_sample: scuba::ScubaSample,
    subset: Option<String>,
}

impl ScubaClientWrapper {
    pub fn log(&self, mut sample: scuba::ScubaSample) {
        sample.join_values(&self.base_sample);
        if let Some(subset) = &self.subset {
            sample.set_subset(subset);
        }
        self.client.log(&sample);
    }

    pub fn flush(&self) {
        match self.client.flush(Duration::from_secs(1)) {
            Err(err) => eprintln!("err while trying to flush scuba client: {err}"),
            _ => {}
        }
    }
}

pub(crate) fn generate_scuba_data(resource: &Resource) -> ScubaClientWrapper {
    // Extract attributes from the OpenTelemetry

    match resource.get(&TABLE) {
        Some(Value::String(table)) => {
            let fb = if was_performed() {
                fbinit::expect_init()
            } else {
                // Safety: This is going to be embedded in a python library, so we can't be sure when fbinit has been called.
                unsafe { fbinit::perform_init() }
            };
            let client = scuba::ScubaClient::new(fb, table.as_str());
            let sample = add_resource_to_sample(resource, scuba::ScubaSample::new());
            let subset = if let Some(Value::String(subset)) = resource.get(&SUBSET) {
                Some(subset.to_string())
            } else {
                None
            };
            ScubaClientWrapper {
                client,
                base_sample: sample,
                subset,
            }
        }
        _ => panic!("you must provide a meta.scuba.table in the given resource"),
    }
}
