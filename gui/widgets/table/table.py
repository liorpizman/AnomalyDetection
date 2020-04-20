#! /usr/bin/env python
#  -*- coding: utf-8 -*-

'''
Anomaly Detection of GPS Spoofing Attacks on UAVs
Authors: Lior Pizman & Yehuda Pashay
GitHub: https://github.com/liorpizman/AnomalyDetection
DataSets: 1. ADS-B dataset 2. simulated data
---
Table widget which is presented in the application
'''

from gui.widgets.table.data_cell import Data_Cell
from gui.widgets.table.header_cell import Header_Cell

try:
    from Tkinter import Frame, Label, Message, StringVar
    from Tkconstants import *
except ImportError:
    from tkinter import Frame, Label, Message, StringVar
    from tkinter.constants import *

try:
    import ttk

    py3 = False
except ImportError:
    import tkinter.ttk as ttk

    py3 = True


class Table(Frame):
    """
     A Class used as a table widget

    Methods
    -------
    _append_n_rows(n)
            Description | Append new n rows

    _pop_n_rows(n)
            Description | Remove n rows

    set_data(data)
            Description | Set the data to the table

    get_data()
            Description | Get the data from the table

    number_of_rows()
            Description | Get the number of rows in the table

    number_of_columns()
            Description | Get the number of columns in the table

    row(index, data=None)
            Description | Handle data change in a specific row

    column(index, data=None)
            Description | Handle data change in a specific column

    clear()
            Description | Clear the table

    delete_row(index)
            Description | Delete a row by a given index

    insert_row(data, index=END)
            Description | Insert a row to the table, can be inserted at a specific index

    cell(row, column, data=None)
            Description | Get the value of a table cell

    __getitem__(index)
            Description | Get the item in a given index

    __setitem__(index, value)
            Description | Set a new value in a given index

    on_change_data(callback):
            Description | Handle changed data with a callback function

    """

    def __init__(self, master, columns, column_weights=None, column_minwidths=None, height=None, minwidth=20,
                 minheight=20, padx=5, pady=5, cell_font=None, cell_foreground="black", cell_background="white",
                 cell_anchor=CENTER, header_font=None, header_background="white", header_foreground="black",
                 header_anchor=CENTER, bordercolor="#999999", innerborder=True, outerborder=True,
                 stripped_rows=("#EEEEEE", "white"), on_change_data=None):
        """
        Parameters
        ----------

        :param master: the master of the table widget
        :param columns: table columns
        :param column_weights: columns width
        :param column_minwidths: column minimum width
        :param height: table height
        :param minwidth: minimum width of the table
        :param minheight: minimum height of the table
        :param padx: x coordinate padding
        :param pady: y coordinate padding
        :param cell_font: the font in the cell
        :param cell_foreground: cell's foreground
        :param cell_background: cell's background
        :param cell_anchor: cell's anchor
        :param header_font: font in header cell
        :param header_background: header cell's background
        :param header_foreground: header cell's foreground
        :param header_anchor: header cell's anchor
        :param bordercolor: the color of the border
        :param innerborder: the inner border
        :param outerborder: the outer border
        :param stripped_rows: whether the rows are stripped or not
        :param on_change_data: behavior on changed data
        """
        outerborder_width = 1 if outerborder else 0

        Frame.__init__(self,
                       master,
                       highlightbackground=bordercolor,
                       highlightcolor=bordercolor,
                       highlightthickness=outerborder_width,
                       bd=0)

        self._cell_background = cell_background
        self._cell_foreground = cell_foreground
        self._cell_font = cell_font
        self._cell_anchor = cell_anchor

        self._number_of_rows = 0
        self._number_of_columns = len(columns)

        self._stripped_rows = stripped_rows

        self._padx = padx
        self._pady = pady

        self._bordercolor = bordercolor
        self._innerborder_width = 1 if innerborder else 0

        self._data_vars = []

        self._columns = columns

        # Handle all header cells according to columns in input
        for j in range(len(columns)):
            column_name = columns[j]

            header_cell = Header_Cell(self, text=column_name, borderwidth=self._innerborder_width, font=header_font,
                                      background=header_background, foreground=header_foreground, padx=padx, pady=pady,
                                      bordercolor=bordercolor, anchor=header_anchor)
            header_cell.grid(row=0, column=j, sticky=N + E + W + S)

        if column_weights is None:
            for j in range(len(columns)):
                self.grid_columnconfigure(j, weight=1)
        else:
            for j, weight in enumerate(column_weights):
                self.grid_columnconfigure(j, weight=weight)

        # Case in which no minimum width for the columns
        if column_minwidths is not None:
            self.update_idletasks()
            for j, minwidth in enumerate(column_minwidths):
                if minwidth is None:
                    header_cell = self.grid_slaves(row=0, column=j)[0]
                    minwidth = header_cell.winfo_reqwidth()
                self.grid_columnconfigure(j, minsize=minwidth)

        if height is not None:
            self._append_n_rows(height)

        self._on_change_data = on_change_data

    def _append_n_rows(self, n):
        """
        Append new n rows
        :param n: number of rows to append
        :return: appended table with n new rows
        """

        number_of_rows = self._number_of_rows
        number_of_columns = self._number_of_columns

        for i in range(number_of_rows + 1, number_of_rows + n + 1):
            list_of_vars = []
            for j in range(number_of_columns):
                var = StringVar()
                list_of_vars.append(var)

                # Handle two different cases - stripped rows or not
                if self._stripped_rows:
                    cell = Data_Cell(self, borderwidth=self._innerborder_width, variable=var,
                                     bordercolor=self._bordercolor, padx=self._padx, pady=self._pady,
                                     background=self._stripped_rows[(i + 1) % 2], foreground=self._cell_foreground,
                                     font=self._cell_font, anchor=self._cell_anchor)
                else:
                    cell = Data_Cell(self, borderwidth=self._innerborder_width, variable=var,
                                     bordercolor=self._bordercolor, padx=self._padx, pady=self._pady,
                                     background=self._cell_background, foreground=self._cell_foreground,
                                     font=self._cell_font, anchor=self._cell_anchor)
                cell.grid(row=i, column=j, sticky=N + E + W + S)

            self._data_vars.append(list_of_vars)

        self._number_of_rows += n

    def _pop_n_rows(self, n):
        """
        Remove n rows
        :param n: n rows to remove
        :return: updated table without n rows
        """

        number_of_rows = self._number_of_rows
        number_of_columns = self._number_of_columns

        # Iterate over table to remove n last rows
        for i in range(number_of_rows - n + 1, number_of_rows + 1):
            for j in range(number_of_columns):
                self.grid_slaves(row=i, column=j)[0].destroy()

            self._data_vars.pop()

        self._number_of_rows -= n

    def set_data(self, data):
        """
        Set the data to the table
        :param data: data to set in the table
        :return: updated table with the new data
        """

        n = len(data)
        m = len(data[0])

        number_of_rows = self._number_of_rows

        if number_of_rows > n:
            self._pop_n_rows(number_of_rows - n)
        elif number_of_rows < n:
            self._append_n_rows(n - number_of_rows)

        for i in range(n):
            for j in range(m):
                self._data_vars[i][j].set(data[i][j])

        if self._on_change_data is not None: self._on_change_data()

    def get_data(self):
        """
        Get the data from the table
        :return: current data in the table
        """
        number_of_rows = self._number_of_rows
        number_of_columns = self._number_of_columns

        data = []

        # Iterate over all the row to collect the data to a list
        for i in range(number_of_rows):
            row = []
            row_of_vars = self._data_vars[i]
            for j in range(number_of_columns):
                cell_data = row_of_vars[j].get()
                row.append(cell_data)

            data.append(row)
        return data

    @property
    def number_of_rows(self):
        """
        Get the number of rows in the table
        :return: number of rows
        """

        return self._number_of_rows

    @property
    def number_of_columns(self):
        """
        Get the number of columns in the table
        :return: number of columns
        """

        return self._number_of_columns

    def row(self, index, data=None):
        """
        Handle data change in a specific row
        :param index: the row index
        :param data: the data in the row
        :return:
        """

        number_of_columns = self._number_of_columns

        # Fill the row when data is not None, otherwise returns the data
        if data is None:
            row = []
            row_of_vars = self._data_vars[index]

            for j in range(number_of_columns):
                row.append(row_of_vars[j].get())

            return row
        else:
            if len(data) != number_of_columns:
                raise ValueError("data has no %d elements: %s" % (number_of_columns, data))

            row_of_vars = self._data_vars[index]
            for j in range(number_of_columns):
                row_of_vars[index][j].set(data[j])

            if self._on_change_data is not None: self._on_change_data()

    def column(self, index, data=None):
        """
        Handle data change in a specific column
        :param index: the column index
        :param data: the data in the column
        :return:
        """

        number_of_rows = self._number_of_rows

        # Fill the column when data is not None, otherwise returns the data
        if data is None:
            column = []

            for i in range(number_of_rows):
                column.append(self._data_vars[i][index].get())

            return column
        else:

            if len(data) != number_of_rows:
                raise ValueError("data has no %d elements: %s" % (number_of_rows, data))

            for i in range(self._number_of_columns):
                self._data_vars[i][index].set(data[i])

            if self._on_change_data is not None: self._on_change_data()

    def clear(self):
        """
        Clear the table
        :return: empty table
        """

        number_of_rows = self._number_of_rows
        number_of_columns = self._number_of_columns

        # Set empty values (strings) in each cell in the table
        for i in range(number_of_rows):
            for j in range(number_of_columns):
                self._data_vars[i][j].set("")

        if self._on_change_data is not None: self._on_change_data()

    def delete_row(self, index):
        """
        Delete a row by a given index
        :param index: the index of the row to delete
        :return: updated table without the row at the input index
        """

        i = index

        # Iterate to a specific index in order to remove the row
        while i < self._number_of_rows:
            row_of_vars_1 = self._data_vars[i]
            row_of_vars_2 = self._data_vars[i + 1]

            j = 0
            while j < self._number_of_columns:
                row_of_vars_1[j].set(row_of_vars_2[j])

            i += 1

        self._pop_n_rows(1)

        if self._on_change_data is not None: self._on_change_data()

    def insert_row(self, data, index=END):
        """
        Insert a row to the table, can be inserted at a specific index
        :param data: data to insert in the new row
        :param index: index of the new row
        :return: updated table with a new row at the input index
        """
        self._append_n_rows(1)

        if index == END:
            index = self._number_of_rows - 1

        i = self._number_of_rows - 1

        # Iterate over rows in order to insert row in a specific index
        # (by default it will be at the end of the table)
        while i > index:
            row_of_vars_1 = self._data_vars[i - 1]
            row_of_vars_2 = self._data_vars[i]

            j = 0
            while j < self._number_of_columns:
                row_of_vars_2[j].set(row_of_vars_1[j])
                j += 1
            i -= 1

        list_of_cell_vars = self._data_vars[index]

        # Iterate over cells in the input row in order to set the data in each cell
        for cell_var, cell_data in zip(list_of_cell_vars, data):
            cell_var.set(cell_data)

        if self._on_change_data is not None: self._on_change_data()

    def cell(self, row, column, data=None):
        """
        Get the value of a table cell
        :param row: cell row
        :param column: cell column
        :param data: the data in the cell if exists
        :return: the data exists in the cell
        """

        if data is None:
            return self._data_vars[row][column].get()
        else:
            self._data_vars[row][column].set(data)
            if self._on_change_data is not None: self._on_change_data()

    def __getitem__(self, index):
        """
        Get the item in a given index
        :param index: input index of an item
        :return: item in a given index
        """

        if isinstance(index, tuple):
            row, column = index
            return self.cell(row, column)
        else:
            raise Exception("Row and column indices are required")

    def __setitem__(self, index, value):
        """
        Set a new value in a given index
        :param index: input index of an item
        :param value: the new value for the item
        :return: updated item
        """

        if isinstance(index, tuple):
            row, column = index
            self.cell(row, column, value)
        else:
            raise Exception("Row and column indices are required")

    def on_change_data(self, callback):
        """
        Handle changed data with a callback function
        :param callback: function to run when the data in the cell was changed
        :return: result of the callback function
        """

        self._on_change_data = callback
