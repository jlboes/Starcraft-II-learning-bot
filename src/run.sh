#!/usr/bin/env bash
export PYTHONPATH=`pwd` && \
export SC2PATH=$PYTHONPATH/../SC2/StarCraftII && \
pysc2_agent --map Simple64 --agent scagent.SmartAgent --agent_race T --max_agent_steps 25000 --screen_resolution 168
