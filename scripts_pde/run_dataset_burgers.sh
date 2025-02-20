
model='gpt4'
opts='evolution_optimize'
data_name=Burgers
# Num=8
# log and result category
catetory='Burgers_seed'

Num=8
seeds="1"
for opt in ${opts}
do
    for seed in ${seeds}
    do
        job_name=${data_name}_${model}_mode=${opt}_num=${Num}_seed=${seed}
        echo ${job_name}
        result_dir=./result/${catetory}
        logdir=./log/${catetory}
        mkdir -p ${result_dir}
        mkdir -p ${logdir}

        python main.py --operators "[+, -, *, /, ^2, ^3]" --operands "[u, u_x, u_xx, u_xxx, x]" \
        --new-add 1 --N ${Num} \
        --optimize-type ${opt} \
        --seed ${seed} \
        --max_epoch 100 \
        --gpt_temperature 0.8 \
        --data-name ${data_name} \
        --job-name ${job_name} --logdir ${result_dir} 
        # > ${logdir}/${job_name}.log 2>&1
    done
done