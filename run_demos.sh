#!/bin/bash

export BERTREND_HOME=$(python -c "import os; import bertrend; print(os.path.dirname(os.path.dirname(bertrend.__file__)))")
supervisord -c supervisord.conf
