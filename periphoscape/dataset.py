class Dataset():
    def __init__(self, data_files):
        self.data_dir = data_files['data_dir']
        self.files = {
            k : f'{data_files["data_dir"]}/{v}'
            for k, v in data_files['files'].items()
        }
        self.number_of_pages = data_files['number_of_pages']
        
