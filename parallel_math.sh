for i in $(seq 1 5);
do
    CMD="bash /home/aiops/liuzc/gem/resources/eval_scripts/eval_math.sh /dataset/liuzc/RLVR-DrGRPO-qwen3-4b-math12k_0726T16:20:45/saved_models/step_00${i}00"
    sailctl job create evalmath${i} -g 1 -p low --debug --image asia-docker.pkg.dev/sail-tpu-02/git-insea-io/common/image_registry/liuzc/nprl:v2 -f values.yaml --command-line-args "$CMD"
    echo $CMD
done