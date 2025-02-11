

### To sync messages and write bag in sim
`rosrun layered_ref_control sync_msg _sim:=true _namespace:=dragonfly1`

  * Bag file gets written to home folder with current data/time as filename

### To sync messages and write bag hw, set the correct namespace based on the MAV ID
`rosrun layered_ref_control sync_msg _sim:=false _namespace:=dragonfly25`