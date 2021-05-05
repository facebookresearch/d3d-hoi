#!/bin/bash
python process_hand_washingmachine.py --start 0 --end 5 --data_path ./final_data_fullv2/washingmachine &
python process_hand_washingmachine.py --start 5 --end 10 --data_path ./final_data_fullv2/washingmachine &
python process_hand_washingmachine.py --start 10 --end 15 --data_path ./final_data_fullv2/washingmachine &
python process_hand_washingmachine.py --start 15 --end 20 --data_path ./final_data_fullv2/washingmachine &
