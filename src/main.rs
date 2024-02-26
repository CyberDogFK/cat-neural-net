use clap::Parser;
use std::error::Error;
use std::io;
use serde::Deserialize;
use rusty_machine::linalg::Matrix;
use rusty_machine::data::transforms::Transformer;
use rusty_machine::data::transforms::Standardizer;
use rusty_machine::learning::nnet::{BCECriterion, NeuralNet};
use rusty_machine::learning::optim::grad_desc::StochasticGD;
use rusty_machine::learning::SupModel;
use rusty_machine::prelude::BaseMatrix;

#[derive(Parser)]
struct Args {
    #[arg(short='r', long="train")]
    /// Training data CSV file
    training_data_csv: std::path::PathBuf,
    #[arg(short='t', long="test")]
    /// Testing data CSV file
    testing_data_csv: std::path::PathBuf,
}

#[derive(Debug, Deserialize)]
struct SampleRow { // [1]
    height: f64,
    length: f64,
    category_id: usize,
}

fn read_data_from_csv(
    file_path: std::path::PathBuf,
) -> Result<(Matrix<f64>, Matrix<f64>), Box<dyn Error>> {
    let mut input_data = vec![];
    let mut label_data = vec![];
    let mut sample_count = 0;
    let mut reader = csv::Reader::from_path(file_path)?; //[2]
    for raw_row in reader.deserialize() { // [3]
        let row: SampleRow = raw_row?;
        input_data.push(row.height);
        input_data.push(row.length);
        label_data.push(row.category_id as f64);
        sample_count += 1;
    }
    let inputs = Matrix::new(sample_count, 2, input_data);
    let targets = Matrix::new(sample_count, 1, label_data);
    Ok((inputs, targets))
}

fn main() -> Result<(), Box<dyn Error>>{
    let options = Args::parse();
    let (training_inputs, training_label_data) = 
        read_data_from_csv(options.training_data_csv)?;
    let mut standardizer = Standardizer::new(0.0, 1.0);
    standardizer.fit(&training_inputs).unwrap();
    let normalized_training_inputs = 
    standardizer.transform(training_inputs).unwrap();
    // ...Train the model with normalzied_training_inputs...
    // Read the testing_inputs
    let (testing_inputs, _) =
    read_data_from_csv(options.testing_data_csv.clone())?;
    // Normalize the testing data with training data
    let normalized_test_cases = standardizer.transform(testing_inputs.clone())?;
    // ...Run the prediction with normalized_test_cases...
    let layers = &[2, 2, 1];
    let criterion = BCECriterion::default();
    let gradient_descent = StochasticGD::new(0.1, 0.1, 20);
    let mut model = NeuralNet::new(
        layers,
        criterion,
        gradient_descent
    );
    model.train(
        &normalized_training_inputs,
        &training_label_data
    )?;
    let (testing_inputs, expected) = read_data_from_csv(
        options.testing_data_csv
    )?;
    // Testing ======================
    // Normalize the testing data using the mean and 
    // variance of the training data
    let res = model.predict(&normalized_test_cases)?;
    let mut writer = csv::Writer::from_writer(io::stdout());
    writer.write_record([
        "height",
        "length",
        "estimated_category_id",
        "true_category_id",
    ])?;
    for row in testing_inputs
        .iter_rows()
        .zip(res.into_vec().into_iter())
        .zip(expected.into_vec().into_iter()) {
        writer.serialize((
            row.0.0[0], row.0.0[1], row.0.1, row.1
            ))?;
    }
    Ok(())
}
