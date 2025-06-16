
CLAN_NAME = 'Skymeadow'


def round_num_to_time_str(round: int) -> str:
	# we set round 1 to be 8:00 AM
	# time increments by 30 minutes each round
	num_30_min_increments = round - 1
	hours = 8 + num_30_min_increments // 2
	minutes = int((num_30_min_increments % 2) * 30)
	
	if hours <= 12:
		am_pm = 'AM'
	else:
		am_pm = 'PM'
		hours -= 12

	return f'{hours}:{minutes:02d} {am_pm}'
