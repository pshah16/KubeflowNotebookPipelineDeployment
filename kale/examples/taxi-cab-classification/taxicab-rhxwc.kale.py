import kfp.dsl as dsl
import kfp.components as comp
from collections import OrderedDict
from kubernetes import client as k8s_client


def data_validation(EPOCHS: int, STEPS: int, BATCH_SIZE: int, HIDDEN_LAYER_SIZE: str, LEARNING_RATE: float):

    import os
    import shutil
    from kale.utils import pod_utils
    from kale.marshal import resource_save as _kale_resource_save
    from kale.marshal import resource_load as _kale_resource_load

    _kale_data_directory = "/marshal"

    if not os.path.isdir(_kale_data_directory):
        os.makedirs(_kale_data_directory, exist_ok=True)

    import os
    import shutil
    import logging
    import apache_beam as beam
    import tensorflow as tf
    import tensorflow_transform as tft
    import tensorflow_model_analysis as tfma
    import tensorflow_data_validation as tfdv

    from apache_beam.io import textio
    from apache_beam.io import tfrecordio

    from tensorflow_transform.beam import impl as beam_impl
    from tensorflow_transform.beam.tft_beam_io import transform_fn_io
    from tensorflow_transform.coders.csv_coder import CsvCoder
    from tensorflow_transform.coders.example_proto_coder import ExampleProtoCoder
    from tensorflow_transform.tf_metadata import dataset_metadata
    from tensorflow_transform.tf_metadata import metadata_io
    DATA_DIR = 'data/'
    TRAIN_DATA = os.path.join(DATA_DIR, 'taxi-cab-classification/train.csv')
    EVALUATION_DATA = os.path.join(
        DATA_DIR, 'taxi-cab-classification/eval.csv')

    # Categorical features are assumed to each have a maximum value in the dataset.
    MAX_CATEGORICAL_FEATURE_VALUES = [24, 31, 12]
    CATEGORICAL_FEATURE_KEYS = ['trip_start_hour',
                                'trip_start_day', 'trip_start_month']

    DENSE_FLOAT_FEATURE_KEYS = ['trip_miles', 'fare', 'trip_seconds']

    # Number of buckets used by tf.transform for encoding each feature.
    FEATURE_BUCKET_COUNT = 10

    BUCKET_FEATURE_KEYS = [
        'pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude']

    # Number of vocabulary terms used for encoding VOCAB_FEATURES by tf.transform
    VOCAB_SIZE = 1000

    # Count of out-of-vocab buckets in which unrecognized VOCAB_FEATURES are hashed.
    OOV_SIZE = 10

    VOCAB_FEATURE_KEYS = ['pickup_census_tract', 'dropoff_census_tract', 'payment_type', 'company',
                          'pickup_community_area', 'dropoff_community_area']

    # allow nan values in these features.
    OPTIONAL_FEATURES = ['dropoff_latitude', 'dropoff_longitude', 'pickup_census_tract', 'dropoff_census_tract',
                         'company', 'trip_seconds', 'dropoff_community_area']

    LABEL_KEY = 'tips'
    FARE_KEY = 'fare'
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
    # tf.get_logger().setLevel(logging.ERROR)

    vldn_output = os.path.join(DATA_DIR, 'validation')

    # TODO: Understand why this was used in the conversion to the output json
    # key columns: list of the names for columns that should be treated as unique keys.
    key_columns = ['trip_start_timestamp']

    # read the first line of the cvs to have and ordered list of column names
    # (the Schema will scrable the features)
    with open(TRAIN_DATA) as f:
        column_names = f.readline().strip().split(',')

    stats = tfdv.generate_statistics_from_csv(data_location=TRAIN_DATA)
    schema = tfdv.infer_schema(stats)

    eval_stats = tfdv.generate_statistics_from_csv(
        data_location=EVALUATION_DATA)
    anomalies = tfdv.validate_statistics(eval_stats, schema)

    # Log anomalies
    for feature_name, anomaly_info in anomalies.anomaly_info.items():
        logging.getLogger().error(
            'Anomaly in feature "{}": {}'.format(
                feature_name, anomaly_info.description))

    # show inferred schema
    tfdv.display_schema(schema=schema)
    # Resolve anomalies
    company = tfdv.get_feature(schema, 'company')
    company.distribution_constraints.min_domain_mass = 0.9

    # Add new value to the domain of feature payment_type.
    payment_type_domain = tfdv.get_domain(schema, 'payment_type')
    payment_type_domain.value.append('Prcard')

    # Validate eval stats after updating the schema
    updated_anomalies = tfdv.validate_statistics(eval_stats, schema)
    tfdv.display_anomalies(updated_anomalies)

    # -----------------------DATA SAVING START---------------------------------
    if "column_names" in locals():
        _kale_resource_save(column_names, os.path.join(
            _kale_data_directory, "column_names"))
    else:
        print("_kale_resource_save: `column_names` not found.")
    if "schema" in locals():
        _kale_resource_save(schema, os.path.join(
            _kale_data_directory, "schema"))
    else:
        print("_kale_resource_save: `schema` not found.")
    # -----------------------DATA SAVING END-----------------------------------


def data_transformation(EPOCHS: int, STEPS: int, BATCH_SIZE: int, HIDDEN_LAYER_SIZE: str, LEARNING_RATE: float):

    import os
    import shutil
    from kale.utils import pod_utils
    from kale.marshal import resource_save as _kale_resource_save
    from kale.marshal import resource_load as _kale_resource_load

    _kale_data_directory = "/marshal"

    if not os.path.isdir(_kale_data_directory):
        os.makedirs(_kale_data_directory, exist_ok=True)

    # -----------------------DATA LOADING START--------------------------------
    _kale_directory_file_names = [
        os.path.splitext(f)[0]
        for f in os.listdir(_kale_data_directory)
        if os.path.isfile(os.path.join(_kale_data_directory, f))
    ]

    if "column_names" not in _kale_directory_file_names:
        raise ValueError("column_names" + " does not exists in directory")

    _kale_load_file_name = [
        f
        for f in os.listdir(_kale_data_directory)
        if os.path.isfile(os.path.join(_kale_data_directory, f)) and
        os.path.splitext(f)[0] == "column_names"
    ]
    if len(_kale_load_file_name) > 1:
        raise ValueError("Found multiple files with name " +
                         "column_names" + ": " + str(_kale_load_file_name))
    _kale_load_file_name = _kale_load_file_name[0]
    column_names = _kale_resource_load(os.path.join(
        _kale_data_directory, _kale_load_file_name))

    if "schema" not in _kale_directory_file_names:
        raise ValueError("schema" + " does not exists in directory")

    _kale_load_file_name = [
        f
        for f in os.listdir(_kale_data_directory)
        if os.path.isfile(os.path.join(_kale_data_directory, f)) and
        os.path.splitext(f)[0] == "schema"
    ]
    if len(_kale_load_file_name) > 1:
        raise ValueError("Found multiple files with name " +
                         "schema" + ": " + str(_kale_load_file_name))
    _kale_load_file_name = _kale_load_file_name[0]
    schema = _kale_resource_load(os.path.join(
        _kale_data_directory, _kale_load_file_name))
    # -----------------------DATA LOADING END----------------------------------

    import os
    import shutil
    import logging
    import apache_beam as beam
    import tensorflow as tf
    import tensorflow_transform as tft
    import tensorflow_model_analysis as tfma
    import tensorflow_data_validation as tfdv

    from apache_beam.io import textio
    from apache_beam.io import tfrecordio

    from tensorflow_transform.beam import impl as beam_impl
    from tensorflow_transform.beam.tft_beam_io import transform_fn_io
    from tensorflow_transform.coders.csv_coder import CsvCoder
    from tensorflow_transform.coders.example_proto_coder import ExampleProtoCoder
    from tensorflow_transform.tf_metadata import dataset_metadata
    from tensorflow_transform.tf_metadata import metadata_io
    DATA_DIR = 'data/'
    TRAIN_DATA = os.path.join(DATA_DIR, 'taxi-cab-classification/train.csv')
    EVALUATION_DATA = os.path.join(
        DATA_DIR, 'taxi-cab-classification/eval.csv')

    # Categorical features are assumed to each have a maximum value in the dataset.
    MAX_CATEGORICAL_FEATURE_VALUES = [24, 31, 12]
    CATEGORICAL_FEATURE_KEYS = ['trip_start_hour',
                                'trip_start_day', 'trip_start_month']

    DENSE_FLOAT_FEATURE_KEYS = ['trip_miles', 'fare', 'trip_seconds']

    # Number of buckets used by tf.transform for encoding each feature.
    FEATURE_BUCKET_COUNT = 10

    BUCKET_FEATURE_KEYS = [
        'pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude']

    # Number of vocabulary terms used for encoding VOCAB_FEATURES by tf.transform
    VOCAB_SIZE = 1000

    # Count of out-of-vocab buckets in which unrecognized VOCAB_FEATURES are hashed.
    OOV_SIZE = 10

    VOCAB_FEATURE_KEYS = ['pickup_census_tract', 'dropoff_census_tract', 'payment_type', 'company',
                          'pickup_community_area', 'dropoff_community_area']

    # allow nan values in these features.
    OPTIONAL_FEATURES = ['dropoff_latitude', 'dropoff_longitude', 'pickup_census_tract', 'dropoff_census_tract',
                         'company', 'trip_seconds', 'dropoff_community_area']

    LABEL_KEY = 'tips'
    FARE_KEY = 'fare'
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
    # tf.get_logger().setLevel(logging.ERROR)

    def to_dense(tensor):
        """Takes as input a SparseTensor and return a Tensor with correct default value
        Args:
          tensor: tf.SparseTensor
        Returns:
          tf.Tensor with default value
        """
        if not isinstance(tensor, tf.sparse.SparseTensor):
            return tensor
        if tensor.dtype == tf.string:
            default_value = ''
        elif tensor.dtype == tf.float32:
            default_value = 0.0
        elif tensor.dtype == tf.int32:
            default_value = 0
        else:
            raise ValueError(f"Tensor type not recognized: {tensor.dtype}")

        return tf.squeeze(tf.sparse_to_dense(tensor.indices,
                                             [tensor.dense_shape[0], 1],
                                             tensor.values, default_value=default_value), axis=1)
        # TODO: Update to below version
        # return tf.squeeze(tf.sparse.to_dense(tensor, default_value=default_value), axis=1)

    def preprocess_fn(inputs):
        """tf.transform's callback function for preprocessing inputs.
        Args:
          inputs: map from feature keys to raw not-yet-transformed features.
        Returns:
          Map from string feature key to transformed feature operations.
        """
        outputs = {}
        for key in DENSE_FLOAT_FEATURE_KEYS:
            # Preserve this feature as a dense float, setting nan's to the mean.
            outputs[key] = tft.scale_to_z_score(to_dense(inputs[key]))

        for key in VOCAB_FEATURE_KEYS:
            # Build a vocabulary for this feature.
            if inputs[key].dtype == tf.string:
                vocab_tensor = to_dense(inputs[key])
            else:
                vocab_tensor = tf.as_string(to_dense(inputs[key]))
            outputs[key] = tft.compute_and_apply_vocabulary(
                vocab_tensor, vocab_filename='vocab_' + key,
                top_k=VOCAB_SIZE, num_oov_buckets=OOV_SIZE)

        for key in BUCKET_FEATURE_KEYS:
            outputs[key] = tft.bucketize(
                to_dense(inputs[key]), FEATURE_BUCKET_COUNT)

        for key in CATEGORICAL_FEATURE_KEYS:
            outputs[key] = tf.cast(to_dense(inputs[key]), tf.int64)

        taxi_fare = to_dense(inputs[FARE_KEY])
        taxi_tip = to_dense(inputs[LABEL_KEY])
        # Test if the tip was > 20% of the fare.
        tip_threshold = tf.multiply(taxi_fare, tf.constant(0.2))
        outputs[LABEL_KEY] = tf.logical_and(
            tf.logical_not(tf.math.is_nan(taxi_fare)),
            tf.greater(taxi_tip, tip_threshold))

        for key in outputs:
            if outputs[key].dtype == tf.bool:
                outputs[key] = tft.compute_and_apply_vocabulary(tf.as_string(outputs[key]),
                                                                vocab_filename='vocab_' + key)

        return outputs
    trns_output = os.path.join(DATA_DIR, "transformed")
    if os.path.exists(trns_output):
        shutil.rmtree(trns_output)

    tft_input_metadata = dataset_metadata.DatasetMetadata(schema)

    runner = 'DirectRunner'
    with beam.Pipeline(runner, options=None) as p:
        with beam_impl.Context(temp_dir=os.path.join(trns_output, 'tmp')):
            converter = CsvCoder(column_names, tft_input_metadata.schema)

            # READ TRAIN DATA
            train_data = (
                p
                | 'ReadTrainData' >> textio.ReadFromText(TRAIN_DATA, skip_header_lines=1)
                | 'DecodeTrainData' >> beam.Map(converter.decode))

            # TRANSFORM TRAIN DATA (and get transform_fn function)
            transformed_dataset, transform_fn = (
                (train_data, tft_input_metadata) | beam_impl.AnalyzeAndTransformDataset(preprocess_fn))
            transformed_data, transformed_metadata = transformed_dataset

            # SAVE TRANSFORMED TRAIN DATA
            _ = transformed_data | 'WriteTrainData' >> tfrecordio.WriteToTFRecord(
                os.path.join(trns_output, 'train'),
                coder=ExampleProtoCoder(transformed_metadata.schema))

            # READ EVAL DATA
            eval_data = (
                p
                | 'ReadEvalData' >> textio.ReadFromText(EVALUATION_DATA, skip_header_lines=1)
                | 'DecodeEvalData' >> beam.Map(converter.decode))

            # TRANSFORM EVAL DATA (using previously created transform_fn function)
            eval_dataset = (eval_data, tft_input_metadata)
            transformed_eval_data, transformed_metadata = (
                (eval_dataset, transform_fn) | beam_impl.TransformDataset())

            # SAVE EVAL DATA
            _ = transformed_eval_data | 'WriteEvalData' >> tfrecordio.WriteToTFRecord(
                os.path.join(trns_output, 'eval'),
                coder=ExampleProtoCoder(transformed_metadata.schema))

            # SAVE transform_fn FUNCTION FOR LATER USE
            # TODO: check out what is the transform function (transform_fn) that came from previous step
            _ = (transform_fn | 'WriteTransformFn' >>
                 transform_fn_io.WriteTransformFn(trns_output))

            # SAVE TRANSFORMED METADATA
            metadata_io.write_metadata(
                metadata=tft_input_metadata,
                path=os.path.join(trns_output, 'metadata'))

    # -----------------------DATA SAVING START---------------------------------
    if "trns_output" in locals():
        _kale_resource_save(trns_output, os.path.join(
            _kale_data_directory, "trns_output"))
    else:
        print("_kale_resource_save: `trns_output` not found.")
    # -----------------------DATA SAVING END-----------------------------------


def train(EPOCHS: int, STEPS: int, BATCH_SIZE: int, HIDDEN_LAYER_SIZE: str, LEARNING_RATE: float):

    import os
    import shutil
    from kale.utils import pod_utils
    from kale.marshal import resource_save as _kale_resource_save
    from kale.marshal import resource_load as _kale_resource_load

    _kale_data_directory = "/marshal"

    if not os.path.isdir(_kale_data_directory):
        os.makedirs(_kale_data_directory, exist_ok=True)

    # -----------------------DATA LOADING START--------------------------------
    _kale_directory_file_names = [
        os.path.splitext(f)[0]
        for f in os.listdir(_kale_data_directory)
        if os.path.isfile(os.path.join(_kale_data_directory, f))
    ]

    if "trns_output" not in _kale_directory_file_names:
        raise ValueError("trns_output" + " does not exists in directory")

    _kale_load_file_name = [
        f
        for f in os.listdir(_kale_data_directory)
        if os.path.isfile(os.path.join(_kale_data_directory, f)) and
        os.path.splitext(f)[0] == "trns_output"
    ]
    if len(_kale_load_file_name) > 1:
        raise ValueError("Found multiple files with name " +
                         "trns_output" + ": " + str(_kale_load_file_name))
    _kale_load_file_name = _kale_load_file_name[0]
    trns_output = _kale_resource_load(os.path.join(
        _kale_data_directory, _kale_load_file_name))
    # -----------------------DATA LOADING END----------------------------------

    import os
    import shutil
    import logging
    import apache_beam as beam
    import tensorflow as tf
    import tensorflow_transform as tft
    import tensorflow_model_analysis as tfma
    import tensorflow_data_validation as tfdv

    from apache_beam.io import textio
    from apache_beam.io import tfrecordio

    from tensorflow_transform.beam import impl as beam_impl
    from tensorflow_transform.beam.tft_beam_io import transform_fn_io
    from tensorflow_transform.coders.csv_coder import CsvCoder
    from tensorflow_transform.coders.example_proto_coder import ExampleProtoCoder
    from tensorflow_transform.tf_metadata import dataset_metadata
    from tensorflow_transform.tf_metadata import metadata_io
    DATA_DIR = 'data/'
    TRAIN_DATA = os.path.join(DATA_DIR, 'taxi-cab-classification/train.csv')
    EVALUATION_DATA = os.path.join(
        DATA_DIR, 'taxi-cab-classification/eval.csv')

    # Categorical features are assumed to each have a maximum value in the dataset.
    MAX_CATEGORICAL_FEATURE_VALUES = [24, 31, 12]
    CATEGORICAL_FEATURE_KEYS = ['trip_start_hour',
                                'trip_start_day', 'trip_start_month']

    DENSE_FLOAT_FEATURE_KEYS = ['trip_miles', 'fare', 'trip_seconds']

    # Number of buckets used by tf.transform for encoding each feature.
    FEATURE_BUCKET_COUNT = 10

    BUCKET_FEATURE_KEYS = [
        'pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude']

    # Number of vocabulary terms used for encoding VOCAB_FEATURES by tf.transform
    VOCAB_SIZE = 1000

    # Count of out-of-vocab buckets in which unrecognized VOCAB_FEATURES are hashed.
    OOV_SIZE = 10

    VOCAB_FEATURE_KEYS = ['pickup_census_tract', 'dropoff_census_tract', 'payment_type', 'company',
                          'pickup_community_area', 'dropoff_community_area']

    # allow nan values in these features.
    OPTIONAL_FEATURES = ['dropoff_latitude', 'dropoff_longitude', 'pickup_census_tract', 'dropoff_census_tract',
                         'company', 'trip_seconds', 'dropoff_community_area']

    LABEL_KEY = 'tips'
    FARE_KEY = 'fare'
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
    # tf.get_logger().setLevel(logging.ERROR)

    def training_input_fn(transformed_output, transformed_examples, batch_size, target_name):
        """
        Args:
          transformed_output: tft.TFTransformOutput
          transformed_examples: Base filename of examples
          batch_size: Batch size.
          target_name: name of the target column.
        Returns:
          The input function for training or eval.
        """
        dataset = tf.data.experimental.make_batched_features_dataset(
            file_pattern=transformed_examples,
            batch_size=batch_size,
            features=transformed_output.transformed_feature_spec(),
            reader=tf.data.TFRecordDataset,
            shuffle=True)
        transformed_features = dataset.make_one_shot_iterator().get_next()
        transformed_labels = transformed_features.pop(target_name)
        return transformed_features, transformed_labels

    def get_feature_columns():
        """Callback that returns a list of feature columns for building a tf.estimator.
        Returns:
          A list of tf.feature_column.
        """
        return (
            [tf.feature_column.numeric_column(key, shape=()) for key in DENSE_FLOAT_FEATURE_KEYS] +
            [tf.feature_column.indicator_column(tf.feature_column.categorical_column_with_identity(key, num_buckets=VOCAB_SIZE + OOV_SIZE)) for key in VOCAB_FEATURE_KEYS] +
            [tf.feature_column.indicator_column(tf.feature_column.categorical_column_with_identity(key, num_buckets=FEATURE_BUCKET_COUNT, default_value=0)) for key in BUCKET_FEATURE_KEYS] +
            [tf.feature_column.indicator_column(tf.feature_column.categorical_column_with_identity(
                key, num_buckets=num_buckets, default_value=0)) for key, num_buckets in zip(CATEGORICAL_FEATURE_KEYS, MAX_CATEGORICAL_FEATURE_VALUES)]
        )
    training_output = os.path.join(DATA_DIR, "training")
    if os.path.exists(training_output):
        shutil.rmtree(training_output)

    hidden_layer_size = [int(x.strip()) for x in HIDDEN_LAYER_SIZE.split(',')]

    tf_transform_output = tft.TFTransformOutput(trns_output)

    # Set how often to run checkpointing in terms of steps.
    config = tf.estimator.RunConfig(save_checkpoints_steps=1000)
    n_classes = tf_transform_output.vocabulary_size_by_name(
        "vocab_" + LABEL_KEY)
    # Create estimator
    estimator = tf.estimator.DNNClassifier(
        feature_columns=get_feature_columns(),
        hidden_units=hidden_layer_size,
        n_classes=n_classes,
        config=config,
        model_dir=training_output)

    # TODO: Simplify all this: https://www.tensorflow.org/guide/premade_estimators
    estimator.train(input_fn=lambda: training_input_fn(
        tf_transform_output,
        os.path.join(trns_output, 'train' + '*'),
        BATCH_SIZE,
        "tips"),
        steps=STEPS)

    # -----------------------DATA SAVING START---------------------------------
    if "tf_transform_output" in locals():
        _kale_resource_save(tf_transform_output, os.path.join(
            _kale_data_directory, "tf_transform_output"))
    else:
        print("_kale_resource_save: `tf_transform_output` not found.")
    if "training_input_fn" in locals():
        _kale_resource_save(training_input_fn, os.path.join(
            _kale_data_directory, "training_input_fn"))
    else:
        print("_kale_resource_save: `training_input_fn` not found.")
    if "estimator" in locals():
        _kale_resource_save(estimator, os.path.join(
            _kale_data_directory, "estimator"))
    else:
        print("_kale_resource_save: `estimator` not found.")
    if "trns_output" in locals():
        _kale_resource_save(trns_output, os.path.join(
            _kale_data_directory, "trns_output"))
    else:
        print("_kale_resource_save: `trns_output` not found.")
    # -----------------------DATA SAVING END-----------------------------------


def eval(EPOCHS: int, STEPS: int, BATCH_SIZE: int, HIDDEN_LAYER_SIZE: str, LEARNING_RATE: float):

    import os
    import shutil
    from kale.utils import pod_utils
    from kale.marshal import resource_save as _kale_resource_save
    from kale.marshal import resource_load as _kale_resource_load

    _kale_data_directory = "/marshal"

    if not os.path.isdir(_kale_data_directory):
        os.makedirs(_kale_data_directory, exist_ok=True)

    # -----------------------DATA LOADING START--------------------------------
    _kale_directory_file_names = [
        os.path.splitext(f)[0]
        for f in os.listdir(_kale_data_directory)
        if os.path.isfile(os.path.join(_kale_data_directory, f))
    ]

    if "trns_output" not in _kale_directory_file_names:
        raise ValueError("trns_output" + " does not exists in directory")

    _kale_load_file_name = [
        f
        for f in os.listdir(_kale_data_directory)
        if os.path.isfile(os.path.join(_kale_data_directory, f)) and
        os.path.splitext(f)[0] == "trns_output"
    ]
    if len(_kale_load_file_name) > 1:
        raise ValueError("Found multiple files with name " +
                         "trns_output" + ": " + str(_kale_load_file_name))
    _kale_load_file_name = _kale_load_file_name[0]
    trns_output = _kale_resource_load(os.path.join(
        _kale_data_directory, _kale_load_file_name))

    if "training_input_fn" not in _kale_directory_file_names:
        raise ValueError("training_input_fn" + " does not exists in directory")

    _kale_load_file_name = [
        f
        for f in os.listdir(_kale_data_directory)
        if os.path.isfile(os.path.join(_kale_data_directory, f)) and
        os.path.splitext(f)[0] == "training_input_fn"
    ]
    if len(_kale_load_file_name) > 1:
        raise ValueError("Found multiple files with name " +
                         "training_input_fn" + ": " + str(_kale_load_file_name))
    _kale_load_file_name = _kale_load_file_name[0]
    training_input_fn = _kale_resource_load(
        os.path.join(_kale_data_directory, _kale_load_file_name))

    if "estimator" not in _kale_directory_file_names:
        raise ValueError("estimator" + " does not exists in directory")

    _kale_load_file_name = [
        f
        for f in os.listdir(_kale_data_directory)
        if os.path.isfile(os.path.join(_kale_data_directory, f)) and
        os.path.splitext(f)[0] == "estimator"
    ]
    if len(_kale_load_file_name) > 1:
        raise ValueError("Found multiple files with name " +
                         "estimator" + ": " + str(_kale_load_file_name))
    _kale_load_file_name = _kale_load_file_name[0]
    estimator = _kale_resource_load(os.path.join(
        _kale_data_directory, _kale_load_file_name))

    if "tf_transform_output" not in _kale_directory_file_names:
        raise ValueError("tf_transform_output" +
                         " does not exists in directory")

    _kale_load_file_name = [
        f
        for f in os.listdir(_kale_data_directory)
        if os.path.isfile(os.path.join(_kale_data_directory, f)) and
        os.path.splitext(f)[0] == "tf_transform_output"
    ]
    if len(_kale_load_file_name) > 1:
        raise ValueError("Found multiple files with name " +
                         "tf_transform_output" + ": " + str(_kale_load_file_name))
    _kale_load_file_name = _kale_load_file_name[0]
    tf_transform_output = _kale_resource_load(
        os.path.join(_kale_data_directory, _kale_load_file_name))
    # -----------------------DATA LOADING END----------------------------------

    import os
    import shutil
    import logging
    import apache_beam as beam
    import tensorflow as tf
    import tensorflow_transform as tft
    import tensorflow_model_analysis as tfma
    import tensorflow_data_validation as tfdv

    from apache_beam.io import textio
    from apache_beam.io import tfrecordio

    from tensorflow_transform.beam import impl as beam_impl
    from tensorflow_transform.beam.tft_beam_io import transform_fn_io
    from tensorflow_transform.coders.csv_coder import CsvCoder
    from tensorflow_transform.coders.example_proto_coder import ExampleProtoCoder
    from tensorflow_transform.tf_metadata import dataset_metadata
    from tensorflow_transform.tf_metadata import metadata_io
    DATA_DIR = 'data/'
    TRAIN_DATA = os.path.join(DATA_DIR, 'taxi-cab-classification/train.csv')
    EVALUATION_DATA = os.path.join(
        DATA_DIR, 'taxi-cab-classification/eval.csv')

    # Categorical features are assumed to each have a maximum value in the dataset.
    MAX_CATEGORICAL_FEATURE_VALUES = [24, 31, 12]
    CATEGORICAL_FEATURE_KEYS = ['trip_start_hour',
                                'trip_start_day', 'trip_start_month']

    DENSE_FLOAT_FEATURE_KEYS = ['trip_miles', 'fare', 'trip_seconds']

    # Number of buckets used by tf.transform for encoding each feature.
    FEATURE_BUCKET_COUNT = 10

    BUCKET_FEATURE_KEYS = [
        'pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude']

    # Number of vocabulary terms used for encoding VOCAB_FEATURES by tf.transform
    VOCAB_SIZE = 1000

    # Count of out-of-vocab buckets in which unrecognized VOCAB_FEATURES are hashed.
    OOV_SIZE = 10

    VOCAB_FEATURE_KEYS = ['pickup_census_tract', 'dropoff_census_tract', 'payment_type', 'company',
                          'pickup_community_area', 'dropoff_community_area']

    # allow nan values in these features.
    OPTIONAL_FEATURES = ['dropoff_latitude', 'dropoff_longitude', 'pickup_census_tract', 'dropoff_census_tract',
                         'company', 'trip_seconds', 'dropoff_community_area']

    LABEL_KEY = 'tips'
    FARE_KEY = 'fare'
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
    # tf.get_logger().setLevel(logging.ERROR)

    eval_result = estimator.evaluate(input_fn=lambda: training_input_fn(
        tf_transform_output,
        os.path.join(trns_output, 'eval' + '*'),
        BATCH_SIZE,
        "tips"),
        steps=50)

    print(eval_result)


data_validation_op = comp.func_to_container_op(
    data_validation, base_image='docker.io/stefanofioravanzo/kale-notebook:0.9')


data_transformation_op = comp.func_to_container_op(
    data_transformation, base_image='docker.io/stefanofioravanzo/kale-notebook:0.9')


train_op = comp.func_to_container_op(
    train, base_image='docker.io/stefanofioravanzo/kale-notebook:0.9')


eval_op = comp.func_to_container_op(
    eval, base_image='docker.io/stefanofioravanzo/kale-notebook:0.9')


@dsl.pipeline(
    name='taxicab-rhxwc',
    description='Use TFX components and Apache Beam to run a ML job over the Chicago taxicab dataset'
)
def auto_generated_pipeline(EPOCHS='1', STEPS='3', BATCH_SIZE='32', HIDDEN_LAYER_SIZE='1500', LEARNING_RATE='0.1'):
    pvolumes_dict = OrderedDict()

    marshal_vop = dsl.VolumeOp(
        name="kale_marshal_volume",
        resource_name="kale-marshal-pvc",
        modes=dsl.VOLUME_MODE_RWM,
        size="1Gi"
    )
    pvolumes_dict['/marshal'] = marshal_vop.volume

    data_validation_task = data_validation_op(EPOCHS, STEPS, BATCH_SIZE, HIDDEN_LAYER_SIZE, LEARNING_RATE)\
        .add_pvolumes(pvolumes_dict)\
        .after()
    data_validation_task.container.working_dir = "/home/jovyan/kale/examples/taxi-cab-classification"
    data_validation_task.container.set_security_context(
        k8s_client.V1SecurityContext(run_as_user=0))

    data_transformation_task = data_transformation_op(EPOCHS, STEPS, BATCH_SIZE, HIDDEN_LAYER_SIZE, LEARNING_RATE)\
        .add_pvolumes(pvolumes_dict)\
        .after(data_validation_task)
    data_transformation_task.container.working_dir = "/home/jovyan/kale/examples/taxi-cab-classification"
    data_transformation_task.container.set_security_context(
        k8s_client.V1SecurityContext(run_as_user=0))

    train_task = train_op(EPOCHS, STEPS, BATCH_SIZE, HIDDEN_LAYER_SIZE, LEARNING_RATE)\
        .add_pvolumes(pvolumes_dict)\
        .after(data_transformation_task)
    train_task.container.working_dir = "/home/jovyan/kale/examples/taxi-cab-classification"
    train_task.container.set_security_context(
        k8s_client.V1SecurityContext(run_as_user=0))

    eval_task = eval_op(EPOCHS, STEPS, BATCH_SIZE, HIDDEN_LAYER_SIZE, LEARNING_RATE)\
        .add_pvolumes(pvolumes_dict)\
        .after(train_task)
    eval_task.container.working_dir = "/home/jovyan/kale/examples/taxi-cab-classification"
    eval_task.container.set_security_context(
        k8s_client.V1SecurityContext(run_as_user=0))


if __name__ == "__main__":
    pipeline_func = auto_generated_pipeline
    pipeline_filename = pipeline_func.__name__ + '.pipeline.tar.gz'
    import kfp.compiler as compiler
    compiler.Compiler().compile(pipeline_func, pipeline_filename)

    # Get or create an experiment and submit a pipeline run
    import kfp
    client = kfp.Client()
    experiment = client.create_experiment('Taxicab')

    # Submit a pipeline run
    run_name = 'taxicab-rhxwc_run'
    run_result = client.run_pipeline(
        experiment.id, run_name, pipeline_filename, {})
