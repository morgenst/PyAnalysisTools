def find_data_period(run_number, data_summary):
    if data_summary is None:
        return None, None
    for year_tag, summary in data_summary.items():
        for period, run_ranges in summary.items():
            for run_range in run_ranges:
                if isinstance(run_range, int):
                    run_range = [run_range]
                if min(run_range) <= run_number <= max(run_range):
                    return period, year_tag
    return None, None
