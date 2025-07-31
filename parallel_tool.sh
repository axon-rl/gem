for i in $(seq 3 5);
do
    CMD="bash /home/aiops/liuzc/gem/resources/eval_scripts/eval_tool.sh /dataset/zichen-qwen3-4b-base-math:Orz57K-py-tool-no-adv-norm-last-line-4env_0723T04:31:43/saved_models/step_00${i}00"
    sailctl job create evalnonorm${i} -g 1 -p low --debug --image asia-docker.pkg.dev/sail-tpu-02/git-insea-io/common/image_registry/liuzc/nprl:v2 -f values.yaml --command-line-args "$CMD"
    echo $CMD
done