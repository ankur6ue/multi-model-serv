
# start redis server on port 6380 because gcs_server listens on 6379
redis-server --port 6380 &
/bin/sleep 1
# Run ray server
RAY_task_events_max_num_task_in_gcs=500 ray start --head --dashboard-host=0.0.0.0 &
# Give Ray a bit of time to come up
/bin/sleep 5
# Start metrics server
python metrics.py &
# Start the deployment graph
serve run generate:generate -a 0.0.0.0:6379 -h 0.0.0.0
# Wait for any process to exit
wait -n
# Exit with status of process that exited first
exit $?
