#!/bin/bash

# Path to the temperature file
TEMP_FILE="/sys/memx0/temperature"
REBOOT_COMMAND="sudo reboot"

while true; do
    # Check if the temperature file exists
    if [[ -f "$TEMP_FILE" ]]; then
        # Read the thermal throttling state from the temperature file
        STATE=$(cat $TEMP_FILE | grep "ThermalThrottlingState")

        # Check if the throttling state indicates a problem (15 in this case)
        if [[ "$STATE" == *"ThermalThrottlingState: 15"* ]]; then
            echo "Thermal throttling detected. Rebooting system."
            $REBOOT_COMMAND
        fi
    else
        echo "Temperature file not found!"
    fi

    sleep 2
done

