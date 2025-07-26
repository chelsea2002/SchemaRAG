db_root_path='./data/bird/database/'
data_mode='dev'
diff_json_path='./data/bird/dev.json'
predicted_sql_path_kg='./log/bird/'
# predicted
ground_truth_path='./data/bird/'
# gold sql
num_cpus=16
meta_time_out=30.0
mode_gt='gt'
mode_predict='gpt'
# error_output_path='./log/..'

echo '''starting to compare with knowledge for ex and ves'''
python -u ./src/evaluation.py --db_root_path ${db_root_path} --predicted_sql_path ${predicted_sql_path_kg} --data_mode ${data_mode} \
--ground_truth_path ${ground_truth_path} --num_cpus ${num_cpus} --mode_gt ${mode_gt} --mode_predict ${mode_predict} \
--diff_json_path ${diff_json_path} --meta_time_out ${meta_time_out} \
# --error_output_path ${error_output_path}

python -u ./src/evaluation_ves.py --db_root_path ${db_root_path} --predicted_sql_path ${predicted_sql_path_kg} --data_mode ${data_mode} \
--ground_truth_path ${ground_truth_path} --num_cpus ${num_cpus} --mode_gt ${mode_gt} --mode_predict ${mode_predict} \
--diff_json_path ${diff_json_path} --meta_time_out ${meta_time_out} 
