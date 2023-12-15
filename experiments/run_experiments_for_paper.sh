#! /bin/bash
#
# Run experiments that reproduce experiments from which published results were
# generated.
#
# The MIT License (MIT)
#
# Copyright © 2023 Honda Research Institute Europe GmbH
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the 'Software'), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED 'AS IS', WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
#
set -e -u -o pipefail

python run_experiments.py --settings ../config/settings_experiments_d_0_0.yml --outpath "../../lbf_experiments/shared_goal_dist_0_0_v8/"
python run_experiments.py --settings ../config/settings_experiments_d_0_2.yml --outpath "../../lbf_experiments/shared_goal_dist_0_2_v8/"
python run_experiments.py --settings ../config/settings_experiments_d_0_5.yml --outpath "../../lbf_experiments/shared_goal_dist_0_5_v8/"
python run_experiments.py --settings ../config/settings_experiments_asym.yml --outpath "../../lbf_experiments/asymmetric_d_0_0_v8/"
