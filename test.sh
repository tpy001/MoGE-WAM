
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export TF_NUM_INTRAOP_THREADS="${TF_NUM_INTRAOP_THREADS:-1}"
export TF_NUM_INTEROP_THREADS="${TF_NUM_INTEROP_THREADS:-1}"
export XLA_FLAGS="${XLA_FLAGS:---xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1}"


CPU_AFFINITY="${CPU_AFFINITY:-48-95}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"


taskset -c "$CPU_AFFINITY" uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi05_latent_flow_depth_simvla_posttrain_libero40 --policy.dir=checkpoints/pi05_latent_flow_depth_simvla_posttrain_libero40/pi05_latent_flow_depth_simvla_posttrain_libero40_reprodeced/29999