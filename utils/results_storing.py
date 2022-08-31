import csv
import yaml

class Storage:
    def __init__(self, config, result_dir, headings, title):
        self.result_dir = result_dir
        self.csv_file = f'{result_dir}/{title}.csv'
        self.config = config
        with open(self.csv_file, 'a') as csvfile:
            filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
            filewriter.writerow(headings)


    def prediction_results(self, results):
        # results = [filename, prediction_character, actual_character, correct_character, prediction_colour, actual_colour, correct_colour]
        with open(self.csv_file, 'a') as csvfile:  # for testing purposes
            filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
            filewriter.writerow(results)

    
    def storage_settings(self):
        with open(self.result_dir / 'setting.yaml', 'w') as f:
            yaml.dump(self.config, f, sort_keys=False)

