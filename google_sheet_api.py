import gspread
from oauth2client.service_account import ServiceAccountCredentials
import pandas as pd

class GoogleSheetAPI():
    def __init__(self, spreadsheet_name):
        #scope = ["https://www.googleapis.com/auth/spreadsheets.readonly"]
        scope = ["https://spreadsheets.google.com/feeds",'https://www.googleapis.com/auth/spreadsheets',"https://www.googleapis.com/auth/drive.file","https://www.googleapis.com/auth/drive"]

        creds = ServiceAccountCredentials.from_json_keyfile_name('astute-anagram-378000-52e17a0d93f4.json', scope)
        client = gspread.authorize(creds)
        self.spreadsheet = client.open(spreadsheet_name)
        print(self.spreadsheet.worksheets)
        self.worksheet = None
    
    def open_worksheet(self, worksheet_id):
        self.worksheet = self.spreadsheet.get_worksheet_by_id(worksheet_id)

    def get_dataframe(self, cell_range, includes_header=False):
        if self.worksheet is None:
            print('You must first open a worksheet')
            return
        data = self.worksheet.get(cell_range)
        dataframe = pd.DataFrame(data)
        if includes_header:
            dataframe.columns = dataframe.iloc[0]
            dataframe = dataframe[1:]

        dataframe = dataframe.apply(pd.to_numeric, errors='coerce')
        return dataframe

    def update_cells(self, cell_range, values):
        if self.worksheet is None:
            print('You must first open a worksheet')
            return
        cell_list = self.worksheet.range(cell_range)
        if len(values) != len(cell_list):
            print('The number of values does not match the number of cells in the range')
            return
        for i, cell in enumerate(cell_list):
            cell.value = values[i]
        self.worksheet.update_cells(cell_list)

    def append_row(self, values):
        if self.worksheet is None:
            print('You must first open a worksheet')
            return
        self.worksheet.append_row(values)

    def update_dataframe(self, dataframe, start_cell='A1', includes_header=False):
        if self.worksheet is None:
            print('You must first open a worksheet')
            return
        if includes_header:
            dataframe = pd.concat([pd.DataFrame([dataframe.columns]), dataframe])
        values = dataframe.values.tolist()
        cell_range = f"{start_cell}:{gspread.utils.rowcol_to_a1(len(values), len(values[0]))}"
        cell_list = self.worksheet.range(cell_range)
        flat_values = [item for sublist in values for item in sublist]
        for i, cell in enumerate(cell_list):
            cell.value = flat_values[i]
        self.worksheet.update_cells(cell_list)