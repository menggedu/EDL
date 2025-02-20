
model='gpt3.5'
opts='evolution_optimize'
data_ids='1'
# 3 4 5 7 8 10 13 16 18 19 20 21 22 23'
data_name_prefix="ODE_"
# log and result category
catetory='ODE_nonlinear'
mode=sparse_regression
Num=10
seeds="1"
metric="0.001"
for opt in ${opts}
do
    for seed in ${seeds}
    do
        for data_id in ${data_ids}
        do 
            data_name=${data_name_prefix}${data_id}
            job_name=${data_name}_${model}_mode=${opt}_num=${Num}_seed=${seed}
            echo ${job_name}
            result_dir=./result/${catetory}
            logdir=./log/${catetory}
            mkdir -p ${result_dir}
            mkdir -p ${logdir}

            python main.py --operators "{+, -, *, /, ^, sin, cos, log, exp}" --operands "{x}" \
            --new-add 1 --N ${Num} \
            --optimize-type ${opt} \
            --seed ${seed} \
            --mode ${mode} \
            --metric_params ${metric} \
            --add-const 1 \
            --max_epoch 100 \
            --data-name ${data_name} \
            --job-name ${job_name} \
            --logdir ${result_dir} > ${logdir}/${job_name}.log 2>&1
        done
    done
done