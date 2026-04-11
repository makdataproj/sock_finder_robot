#!/bin/bash
#!/bin/bash

SESSION="nav2_stack"
BASE_DIR="$HOME/nav2_dir"

tmux has-session -t "$SESSION" 2>/dev/null
if [ $? -eq 0 ]; then
    echo "Session '$SESSION' already exists."
    echo "Attach with: tmux attach -t $SESSION"
    exit 0
fi

tmux new-session -d -s "$SESSION" -n base "cd '$BASE_DIR' && ./bringup.sh"

tmux new-window -t "$SESSION" -n nav2 "cd '$BASE_DIR' && sleep 3 && ./nav2_launch.sh"
tmux new-window -t "$SESSION" -n localization "cd '$BASE_DIR' && sleep 6 && ./localization.sh"
#tmux new-window -t "$SESSION" -n rviz "cd '$BASE_DIR' && sleep 9 && ./nav_rviz.sh"

tmux select-window -t "$SESSION":0
tmux attach -t "$SESSION"
