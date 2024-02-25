use clap::Parser;
use ndarray::Array2;
use rand::distributions::Distribution; // for using .sample()
use rand::thread_rng;
use rand_distr::Normal; // split from rand since 0.7
use serde::Deserialize;
use serde::Serialize;
use std::fs::read_to_string;
use std::io;
use std::error::Error;

#[derive(Deserialize)]
struct Config {
    centroids: [f64; 4],
    noise: f64,
    samples_per_centroid: usize,
}

#[derive(Debug, Serialize)]
struct Sample {
    // [1]
    height: f64,
    length: f64,
    category_id: usize,
}

fn generate_data(
    centroids: &Array2<f64>,
    points_per_centroid: usize,
    noise: f64
) -> Vec<Sample> {
    assert!(
        !centroids.is_empty(),
        "centroids cannot be empty."
    );
    assert!(noise >= 0f64, "noise mut be non-negative.");
    let cols = centroids.shape()[1];
    let mut rng = thread_rng();
    let normal_rv = Normal::new(0f64, noise).unwrap();
    let mut samples = Vec::with_capacity(points_per_centroid);
    for _ in 0..points_per_centroid {
        // generate points from each centroid
        for (centroid_id, centroid) in centroids
            .rows()
            .into_iter()
            .enumerate() {
            // generate a point randomly around the centroid
            let mut point = Vec::with_capacity(cols);
            for feature in centroid.into_iter() {
                point.push(feature + normal_rv.sample(&mut rng));
            }
            samples.push(Sample {
                height: point[0],
                length: point[1],
                category_id: centroid_id,
            });
        }
    };
    samples
}

#[derive(Parser)]
struct Args {
    #[arg(short='c', long="config-file")]
    config_file_path: std::path::PathBuf,
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();
    let toml_config_str = read_to_string(args.config_file_path)?;
    let config: Config = toml::from_str(
        &toml_config_str
    ).unwrap();
    let centroids = Array2::from_shape_vec(
        (2, 2),
        config.centroids.to_vec()
    )?;
    let samples = generate_data(
        &centroids,
        config.samples_per_centroid,
        config.noise
    );
    let mut writer = csv::Writer::from_writer(io::stdout());
    for sample in samples {
        writer.serialize(sample)?;
    }
    Ok(())
}