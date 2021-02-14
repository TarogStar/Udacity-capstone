# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved. Modified
# ---------------------------------------------------------
import json
import logging
import os
import pickle
import numpy as np
import pandas as pd
import joblib

import azureml.automl.core
from azureml.automl.core.shared import logging_utilities, log_server
from azureml.telemetry import INSTRUMENTATION_KEY

from inference_schema.schema_decorators import input_schema, output_schema
from inference_schema.parameter_types.numpy_parameter_type import NumpyParameterType
from inference_schema.parameter_types.pandas_parameter_type import PandasParameterType


input_sample = pd.DataFrame({"app.cre_AppId": pd.Series([0], dtype="int64"), "app.cre_Experience": pd.Series([0.0], dtype="float64"), "app.cre_MonthsExperienceinPast36": pd.Series([0.0], dtype="float64"), "app.cre_PardotScore": pd.Series(["example_value"], dtype="object"), "app.cre_Veteran": pd.Series([0.0], dtype="float64"), "app.cre_WantTeamDriver": pd.Series([0.0], dtype="float64"), "app.cre_DriverApplicationSource": pd.Series(["example_value"], dtype="object"), "app.cre_RecordSource": pd.Series(["example_value"], dtype="object"), "app.cre_CDLType": pd.Series([0.0], dtype="float64"), "app.cre_AccidentInformationProvided": pd.Series([False], dtype="bool"), "app.cre_ContactInformationProvided": pd.Series([False], dtype="bool"), "app.cre_CriminalInformationProvided": pd.Series([False], dtype="bool"), "app.cre_TicketInformationProvided": pd.Series([False], dtype="bool"), "app.cre_ScoreCurrent": pd.Series([0.0], dtype="float64"), "app.cre_ScoreInitial": pd.Series([0.0], dtype="float64"), "app.cre_VettingStatus": pd.Series([0.0], dtype="float64"), "app.cre_AccidentCount": pd.Series([0.0], dtype="float64"), "app.cre_DUICount": pd.Series([0.0], dtype="float64"), "app.cre_MovingViolationCount": pd.Series([0.0], dtype="float64"), "app.cre_SoftFicoScore": pd.Series([0.0], dtype="float64"), "app.cre_CDLCLPExp": pd.Series(["example_value"], dtype="object"), "app.cre_FelonyCount": pd.Series([0.0], dtype="float64"), "address1_postalcode": pd.Series([0.0], dtype="float64"), "cre_referralcode": pd.Series(["example_value"], dtype="object"), "cre_referralestimatedexperience": pd.Series([0.0], dtype="float64"), "cre_referralsourceid": pd.Series(["example_value"], dtype="object"), "cre_accidentcount": pd.Series([0.0], dtype="float64"), "cre_canpassdrugtest": pd.Series(["example_value"], dtype="object"), "cre_cdlclass": pd.Series(["example_value"], dtype="object"), "cre_cdlexp": pd.Series([0.0], dtype="float64"), "cre_duicount": pd.Series(["example_value"], dtype="object"), "cre_hascdl": pd.Series(["example_value"], dtype="object"), "cre_honorablydischarged": pd.Series(["example_value"], dtype="object"), "cre_movingviolationcount": pd.Series([0.0], dtype="float64"), "cre_recordsource": pd.Series([0.0], dtype="float64"), "cre_veteran": pd.Series(["example_value"], dtype="object"), "cre_washonorablydischarged": pd.Series([0.0], dtype="float64"), "cre_minsoftficoscore": pd.Series(["example_value"], dtype="object"), "cre_softficoscore": pd.Series(["example_value"], dtype="object"), "cre_militarydischargedon": pd.Series(["2000-1-1"], dtype="datetime64[ns]"), "cre_recklessdrivingcount": pd.Series([0.0], dtype="float64"), "cre_driverchewtobacco": pd.Series([0.0], dtype="float64"), "cre_driversmoker": pd.Series([0.0], dtype="float64"), "cre_drivervapeuser": pd.Series([0.0], dtype="float64"), "cre_teamchewtobaccousers": pd.Series([0.0], dtype="float64"), "cre_teamoppositegender": pd.Series(["example_value"], dtype="object"), "cre_teamsmokers": pd.Series([0.0], dtype="float64"), "cre_teamvapeusers": pd.Series([0.0], dtype="float64"), "cre_teamgender": pd.Series([0.0], dtype="float64"), "cre_donottext": pd.Series([False], dtype="bool")})
output_sample = np.array([0])
try:
    log_server.enable_telemetry(INSTRUMENTATION_KEY)
    log_server.set_verbosity('INFO')
    logger = logging.getLogger('azureml.automl.core.scoring_script')
except:
    pass


def init():
    global model
    # This name is model.id of model that we want to deploy deserialize the model file back
    # into a sklearn model
    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'model.pkl')
    path = os.path.normpath(model_path)
    path_split = path.split(os.sep)
    log_server.update_custom_dimensions({'model_name': path_split[1], 'model_version': path_split[2]})
    try:
        logger.info("Loading model from path.")
        model = joblib.load(model_path)
        logger.info("Loading successful.")
    except Exception as e:
        logging_utilities.log_traceback(e, logger)
        raise


@input_schema('data', PandasParameterType(input_sample))
@output_schema(NumpyParameterType(output_sample))
def run(data):
    try:
        result = model.predict(data)
        return json.dumps({"result": result.tolist()})
    except Exception as e:
        result = str(e)
        return json.dumps({"error": result})
