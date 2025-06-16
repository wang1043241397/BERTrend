#
# Copyright (c) 2024, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# SPDX-License-Identifier: MPL-2.0
# This file is part of BERTrend.
#


# Set logs directory and create if not exists
export BERTREND_LOGS_DIR=$BERTREND_BASE_DIR/logs/bertrend
mkdir -p $BERTREND_LOGS_DIR

echo "Starting BERTrend Topic Analysis demo"
cd `pwd`/topic_analysis && CUDA_VISIBLE_DEVICES=0 streamlit run app.py 2>&1 | tee -a $BERTREND_LOGS_DIR/topic_analysis_demo.log
