use std::env;
use std::path::Path;

fn main() {
    let sources = [Path::new("src/resource.fbs"), Path::new("src/404.json")];
    let out_dir = env::var("OUT_DIR").unwrap();

    for source in &sources {
        println!("cargo:rerun-if-changed={}", source.display());
    }

    flatc_rust::run(flatc_rust::Args {
        inputs: &sources,
        binary: true,
        out_dir: Path::new(&out_dir),
        ..Default::default()
    })
    .expect("flatc");
}
