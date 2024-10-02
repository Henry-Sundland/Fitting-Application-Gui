import tkinter as tk
from tkinter import filedialog, ttk # ttk is for dropdown menus
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np


class MainGUI:

    def __init__(self, master):
        self.master = master
        self.master.title("Henry Sundland's Cool Data Fitting Application, Unifinished Prototype (In Progress Though :) )")
        self.master.state('zoomed')  # Maximize the window

        # Variables to store loaded data
        self.data = None
        self.x_col_data = None
        self.y_col_data = None
        self.x_uncertainties = None
        self.y_uncertainties = None

        # Create the buttons and interface
        self.load_button = tk.Button(master, text="Load Data", command=self.load_data)
        self.load_button.pack(pady=10)

        self.data_label = tk.Label(master, text="No data loaded.")
        self.data_label.pack(pady=10)

        # Dropdowns for selecting X and Y
        self.x_column = tk.StringVar()
        self.y_column = tk.StringVar()
        self.x_uncert_column = tk.StringVar()
        self.y_uncert_column = tk.StringVar()

        self.x_dropdown = None
        self.y_dropdown = None
        self.x_uncert_dropdown = None
        self.y_uncert_dropdown = None
        self.confirm_button = None
        self.plot_widget = None  # For storing the current plot



        # Add the fit type dropdown
        self.fit_type = tk.StringVar()


        # Add polynomial order entry (but make it appear only when Polynomial Fit is selected)
        self.poly_order_label = tk.Label(master, text="Polynomial Order:")
        self.poly_order_entry = tk.Entry(master)
        self.poly_order_label.place_forget()  # Hide initially
        self.poly_order_entry.place_forget()  # Hide initially   






        fit_options = ['Linear Fit (no uncertainties)','Linear Fit (with y uncertainties)', 'Power-Law Fit (with uncertainties)', 'Polynomial Fit']  # Add your desired fit options here
        self.fit_dropdown = ttk.Combobox(master, textvariable=self.fit_type, values=fit_options)
        self.fit_dropdown.place(relx=0.90, rely=0.5, anchor='center')  # Position it in the middle-far right
        self.fit_dropdown.set('Select Fit Type')

        #Link the dropdown selection to show/hide the polynomial order input
        self.fit_dropdown.bind("<<ComboboxSelected>>", self.on_fit_selection)        


        # Fit Parameters Label
        self.fit_params_label = tk.Label(master, text="")
        self.fit_params_label.place(relx=0.88, rely=0.70, anchor='center')

        # Add a "Fit!" button to apply the fit
        self.fit_button = tk.Button(master, text="Fit!", command=self.apply_fit)
        self.fit_button.place(relx=0.90, rely=0.62, anchor='center')


        # Add the Exit Program button and place it towards the bottom left
        self.exit_button = tk.Button(master, text="Exit Program", command=self.master.quit)
        self.exit_button.place(relx=0.01, rely=0.95, anchor='sw')  # Adjust the button's position



    def on_fit_selection(self, event):
        # If 'Polynomial Fit' is selected, show the polynomial order input
        fit_type = self.fit_type.get()
        if fit_type == 'Polynomial Fit':
            self.poly_order_label.place(relx=0.85, rely=0.57, anchor='center')
            self.poly_order_entry.place(relx=0.90, rely=0.57, anchor='center')
        else:
            self.poly_order_label.place_forget()
            self.poly_order_entry.place_forget()





    def apply_fit(self):
        # Get the selected fit type
        fit_type = self.fit_dropdown.get()


        if fit_type == 'Polynomial Fit':
            poly_order = int(self.poly_order_entry.get())
            fitter = FittingClass(self.x_col_data, self.y_col_data, self.y_uncertainties)
            coefficients, uncertainties, chi_squared = fitter.polynomial_fit_with_uncertainties(poly_order)
            self.plot_polynomial_fit(coefficients)
            self.fit_params_label.config(text=f"Polynomial Coefficients: {coefficients}\n"
                                              f"Uncertainties: {uncertainties}\n"
                                              f"Chi²: {chi_squared:.2f}")



        elif fit_type == 'Linear Fit (no uncertainties)':
            # Pass y_uncertainties as None for no-uncertainties fit
            fitter = FittingClass(self.x_col_data, self.y_col_data, None)
            slope, y_intercept, slope_uncert, y_intercept_uncert = fitter.linear_fit_no_uncertainties()
            self.plot_with_fit(slope, y_intercept)
            self.fit_params_label.config(text=f"Slope: {slope:.4f} ± {slope_uncert:.4f}, Intercept: {y_intercept:.4f} ± {y_intercept_uncert:.4f}")

        elif fit_type == 'Linear Fit (with y uncertainties)':
            # Pass the actual uncertainties when needed
            fitter = FittingClass(self.x_col_data, self.y_col_data, self.y_uncertainties)
            slope, y_intercept, slope_uncert, y_intercept_uncert, chi_squared = fitter.linear_fit_with_uncertainties()
            self.plot_with_fit(slope, y_intercept)
            self.fit_params_label.config(text=f"Slope: {slope:.4f} ± {slope_uncert:.4f}\n"
                                 f"Intercept: {y_intercept:.4f} ± {y_intercept_uncert:.4f}\n"
                                 f"Chi²: {chi_squared:.2f}")
            

        elif fit_type == 'Power-Law Fit (with uncertainties)':
            fitter = FittingClass(self.x_col_data, self.y_col_data, self.y_uncertainties)
            a_fit, b_fit, uncertainty_a_fit, uncertainty_b_fit, chi_squared = fitter.power_law_fit_with_uncertainties()
            self.plot_power_fit(a_fit, b_fit)
            self.fit_params_label.config(text=f"a (coefficient): {a_fit:.4f} ± {uncertainty_a_fit:.4f}\n"
                                            f"b (exponent): {b_fit:.4f} ± {uncertainty_b_fit:.4f}\n"
                                            f"Chi²: {chi_squared:.2f}")



    def plot_polynomial_fit(self, coefficients):
        if self.plot_widget:
            self.plot_widget.get_tk_widget().destroy()

        fig, ax = plt.subplots(figsize=(7, 6))

        if self.y_uncertainties is not None:
            ax.errorbar(self.x_col_data, self.y_col_data, yerr=self.y_uncertainties, fmt='o', color='blue', ecolor='gray', capsize=5)
        else:
            ax.scatter(self.x_col_data, self.y_col_data, label='Data', color='blue')

        y_fit = np.polyval(coefficients, self.x_col_data)
        ax.plot(self.x_col_data, y_fit, label='Polynomial Fit', color='red')

        ax.legend()
        ax.set_title(f'Polynomial Fit (Degree {len(coefficients)-1})')
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')

        self.plot_widget = FigureCanvasTkAgg(fig, master=self.master)
        self.plot_widget.draw()
        self.plot_widget.get_tk_widget().pack(pady=10)



    def plot_with_fit(self, slope, y_intercept):
        # Clear the old plot
        if self.plot_widget:
            self.plot_widget.get_tk_widget().destroy()

        # Create a new figure for the plot with the fit line
        fig, ax = plt.subplots(figsize=(7, 6))

        # Check if uncertainties are provided (i.e., y_uncertainties is not None)
        if self.y_uncertainties is not None:
            # Plot the original data points with error bars
            ax.errorbar(self.x_col_data, self.y_col_data, yerr=self.y_uncertainties, fmt='o', label='Data with uncertainties', color='blue', ecolor='gray', capsize=5)
        else:
            # Plot the original data points without uncertainties
            ax.scatter(self.x_col_data, self.y_col_data, label='Data', color='blue')

        # Plot the fitted line
        y_fit = slope * self.x_col_data + y_intercept
        ax.plot(self.x_col_data, y_fit, label=f'Fit: y = {slope:.2f}x + {y_intercept:.2f}', color='red')

        # Add legend and title
        ax.legend()
        ax.set_title('Fitted Linear Line')
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')

        # Embed the plot into the Tkinter window
        self.plot_widget = FigureCanvasTkAgg(fig, master=self.master)
        self.plot_widget.draw()
        self.plot_widget.get_tk_widget().pack(pady=10)


    def plot_power_fit(self, a_fit, b_fit):
        # Clear the old plot
        if self.plot_widget:
            self.plot_widget.get_tk_widget().destroy()

        # Create a new figure for the plot
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot the original data points with error bars if uncertainties are provided
        if self.y_uncertainties is not None:
            ax.errorbar(self.x_col_data, self.y_col_data, yerr=self.y_uncertainties, fmt='o', 
                        label='Data with uncertainties', color='blue', ecolor='gray', capsize=5)
        else:
            ax.scatter(self.x_col_data, self.y_col_data, label='Data', color='blue')

        # Plot the power-law fit
        y_fit = a_fit * self.x_col_data**b_fit
        ax.plot(self.x_col_data, y_fit, label=f'Fit: y = {a_fit:.2f} * x^{b_fit:.2f}', color='red')

        # Add legend, labels, and title
        ax.legend()
        ax.set_title('Power-Law Fit')
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')

        # Embed the plot into the Tkinter window
        self.plot_widget = FigureCanvasTkAgg(fig, master=self.master)
        self.plot_widget.draw()
        self.plot_widget.get_tk_widget().pack(pady=10)

    def display_polynomial_fit_params(self, coefficients, uncertainties):
        # Arrange coefficients and uncertainties into rows of 6
        display_text = ""
        for i in range(0, len(coefficients), 6):
            # Get a slice of 6 coefficients and their uncertainties for this row
            coeff_slice = coefficients[i:i+6]
            uncert_slice = uncertainties[i:i+6]

            # Format the row
            row_text = ""
            for coeff, uncert in zip(coeff_slice, uncert_slice):
                row_text += f"{coeff:.4f} ± {uncert:.4f}    "

            # Add the row to the display text
            display_text += row_text + "\n"

        # Set the display text on the fit_params_label
        self.fit_params_label.config(text=display_text)


    def load_data(self):
        # Ask for either an Excel or a pickle file
        file_path = filedialog.askopenfilename(filetypes=[("Excel files", "*.xlsx"), ("Pickle files", "*.pkl")])

        if file_path.endswith('.xlsx'):
            self.data = pd.read_excel(file_path)  # Load Excel file
        elif file_path.endswith('.pkl'):
            self.data = pd.read_pickle(file_path)  # Load pickle file

        if self.data is not None:
            self.data_label.config(text=f"Loaded: {file_path}")
            self.clear_old_widgets()  # Clear previous dropdowns before showing new ones
            self.show_column_selection()

    def clear_old_widgets(self):
        # Destroy old dropdown menus and buttons if they exist
        if self.x_dropdown:
            self.x_dropdown.destroy()
        if self.y_dropdown:
            self.y_dropdown.destroy()
        if self.x_uncert_dropdown:
            self.x_uncert_dropdown.destroy()
        if self.y_uncert_dropdown:
            self.y_uncert_dropdown.destroy()
        if self.confirm_button:
            self.confirm_button.destroy()
        
        if self.plot_widget:
            # Destroy the previous plot widget to make space for a new one
            self.plot_widget.get_tk_widget().destroy()
            self.plot_widget = None  # Ensure no residual reference remains

    def show_column_selection(self):
        # Get the column names from the loaded data
        columns = list(self.data.columns)

        # Add "None" as an option for uncertainties
        
        

        columns_with_none = ['None'] + columns

        # Create dropdowns for selecting X and Y columns
        self.x_dropdown = ttk.Combobox(self.master, textvariable=self.x_column, values=columns)
        self.x_dropdown.pack(pady=5)
        self.x_dropdown.set('Select X Column')

        self.y_dropdown = ttk.Combobox(self.master, textvariable=self.y_column, values=columns)
        self.y_dropdown.pack(pady=5)
        self.y_dropdown.set('Select Y Column')

        self.x_uncert_dropdown = ttk.Combobox(self.master, textvariable=self.x_uncert_column, values=columns_with_none)
        self.x_uncert_dropdown.pack(pady=5)
        self.x_uncert_dropdown.set('Select X Uncertainty Column (Optional)')

        self.y_uncert_dropdown = ttk.Combobox(self.master, textvariable=self.y_uncert_column, values=columns_with_none)
        self.y_uncert_dropdown.pack(pady=5)
        self.y_uncert_dropdown.set('Select Y Uncertainty Column (Optional)')

        # Add a button to confirm selection
        self.confirm_button = tk.Button(self.master, text="Confirm Selection", command=self.confirm_selection)
        self.confirm_button.pack(pady=5)

    def confirm_selection(self):
        # Get the selected X, Y, and Uncertainty columns
        x_col = self.x_column.get()
        y_col = self.y_column.get()
        x_uncert_col = self.x_uncert_column.get()
        y_uncert_col = self.y_uncert_column.get()

        # Ensure valid column selections for X and Y
        if x_col and y_col and x_col != 'Select X Column' and y_col != 'Select Y Column':
            self.x_col_data = self.data[x_col]
            self.y_col_data = self.data[y_col]
            
            # If the user selected "None" for X Uncertainty or no valid column, set to None
            if x_uncert_col and x_uncert_col != 'Select X Uncertainty Column (Optional)' and x_uncert_col != 'None':
                self.x_uncertainties = self.data[x_uncert_col]
            else:
                self.x_uncertainties = None

            # If the user selected "None" for Y Uncertainty or no valid column, set to None
            if y_uncert_col and y_uncert_col != 'Select Y Uncertainty Column (Optional)' and y_uncert_col != 'None':
                self.y_uncertainties = self.data[y_uncert_col]
            else:
                self.y_uncertainties = None

            # Notify user that data is ready for fitting
            self.clear_old_widgets()  # Clear dropdowns after confirming selection
            # Call the plot function to plot the selected data
            self.plot_data(x_col, y_col)  # This will trigger the plot
            self.data_label.config(text=f"Data ready for fitting (X: {x_col}, Y: {y_col}, X Uncertainty: {x_uncert_col if x_uncert_col != 'None' else 'None'}, Y Uncertainty: {y_uncert_col if y_uncert_col != 'None' else 'None'})")
        else:
            self.data_label.config(text="Please select valid columns for both X and Y.")

    def plot_data(self, x_col, y_col):
        # create a figure for the plot
        fig, ax = plt.subplots(figsize = (9,6))

        # Check if uncertainties are provided
        if self.x_uncertainties is None and self.y_uncertainties is None:
            # Scatter plot without uncertainties
            ax.scatter(self.x_col_data, self.y_col_data, label=f'{x_col} vs {y_col}', color='b')
        else:
            # If uncertainties are provided, plot with error bars
            xerr = self.x_uncertainties if self.x_uncertainties is not None else None
            yerr = self.y_uncertainties if self.y_uncertainties is not None else None
            ax.errorbar(self.x_col_data, self.y_col_data, xerr=xerr, yerr=yerr, fmt='o', label=f'{x_col} vs {y_col}', color='b')

        # Add legend and title
        ax.legend()
        ax.set_title(f'{x_col} vs {y_col}')
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)

        # Embed the plot into the Tkinter window
        self.plot_widget = FigureCanvasTkAgg(fig, master=self.master)
        self.plot_widget.draw()
        self.plot_widget.get_tk_widget().pack(pady=10)


class FittingClass:

    def __init__(self, x_data, y_data, y_uncertainties=None):
        self.x_data = x_data
        self.y_data = y_data
        self.y_uncertainties = y_uncertainties  # This is the fix, properly assigning y_uncertainties


    def power_law_fit_with_uncertainties(self):
        x = self.x_data
        y = self.y_data
        dy = self.y_uncertainties  # Uncertainties in y

        # Linearize the data by taking the logarithm
        X = np.log(x)
        Y = np.log(y)
        Y_uncertainty = dy / y
        weights = 1 / (Y_uncertainty ** 2)

        # Weighted least-squares fit
        weight_sum = np.sum(weights)
        S_x = np.sum(weights * X)
        S_xx = np.sum(weights * X**2)
        S_xy = np.sum(weights * X * Y)
        S_y = np.sum(weights * Y)

        # Calculate fit parameters A and B
        delta = S_xx * weight_sum - (S_x ** 2)
        A = (weight_sum * S_xy - S_x * S_y) / delta
        B = (S_xx * S_y - S_x * S_xy) / delta
        A_uncertainty = np.sqrt(weight_sum / delta)
        B_uncertainty = np.sqrt(S_xx / delta)

        # Convert back to the original space
        a_fit = np.exp(B)
        b_fit = A
        uncertainty_a_fit = a_fit * B_uncertainty
        uncertainty_b_fit = A_uncertainty

        # Recalculate the fit in the original space
        y_fit = a_fit * x**b_fit

        residuals = y - y_fit  # Residuals in original space
        chi_squared = np.sum((residuals / dy) ** 2)  # Chi-squared in original space


        return a_fit, b_fit, uncertainty_a_fit, uncertainty_b_fit, chi_squared



    def polynomial_fit_with_uncertainties(self, degree):
        x = self.x_data
        y = self.y_data
        dy = self.y_uncertainties

        if dy is None:
            # Simple polynomial fit without uncertainties
            coefficients = np.polyfit(x, y, degree)
            y_fit = np.polyval(coefficients, x)
            residuals = y - y_fit
            chi_squared = np.sum(residuals ** 2)
            uncertainties = np.zeros_like(coefficients)  # Placeholder; uncertainties require more complex calculations

        else:
            # Weighted polynomial fit
            weights = 1 / dy**2
            coefficients = np.polyfit(x, y, degree, w=weights)
            y_fit = np.polyval(coefficients, x)
            residuals = (y - y_fit) / dy
            chi_squared = np.sum(residuals ** 2)
            uncertainties = np.sqrt(np.diag(np.linalg.inv(np.dot(np.vander(x, degree+1).T, np.vander(x, degree+1)))))  # Approximate uncertainties

        return coefficients, uncertainties, chi_squared




    def linear_fit_no_uncertainties(self):
        # perform the linear least squares fitting

        x = self.x_data
        y = self.y_data
        N = len(x)

        # Linear least squares fitting calculation
        x_sum = sum(x)
        y_sum = sum(y)
        x_squared_sum = sum(x**2)
        x_y_sum = sum(x*y)

        slope = (N*x_y_sum - x_sum*y_sum)/(N*x_squared_sum - x_sum**2)
        y_intercept = (y_sum - slope*x_sum)/N

        # predict y values for the fit line
        y_fit = slope*x + y_intercept


        # Calculate residuals and uncertainties
        residuals = y - (slope * x + y_intercept)
        residual_sum_of_squares = sum(residuals**2)
        sigma_squared = residual_sum_of_squares / (N - 2)
        slope_uncertainty = np.sqrt(N * sigma_squared / (N * x_squared_sum - x_sum**2))
        y_intercept_uncertainty = np.sqrt(sigma_squared * x_squared_sum / (N * (N * x_squared_sum - x_sum**2)))


        return slope, y_intercept, slope_uncertainty, y_intercept_uncertainty

    def linear_fit_with_uncertainties(self):
        x = self.x_data
        y = self.y_data
        dy = self.y_uncertainties  # Uncertainties in y

        # Weighted fit using uncertainties
        weights = 1 / (dy**2)
        weight_sum = np.sum(weights)
        S_x = np.sum(weights * x)
        S_xx = np.sum(weights * (x**2))
        S_xy = np.sum(weights * x * y)
        S_y = np.sum(weights * y)

        # Fit parameters (slope and intercept)
        delta = S_xx * weight_sum - (S_x**2)
        slope = (weight_sum * S_xy - S_x * S_y) / delta
        y_intercept = (S_xx * S_y - S_x * S_xy) / delta

        # Uncertainties in the fit parameters
        slope_uncertainty = np.sqrt(weight_sum / delta)
        y_intercept_uncertainty = np.sqrt(S_xx / delta)

        # Calculate residuals and chi-squared
        y_fit = slope * x + y_intercept  # Fitted y-values
        residuals = y - y_fit
        chi_squared = np.sum((residuals / dy) ** 2)  # Correct Chi-squared calculation

        return slope, y_intercept, slope_uncertainty, y_intercept_uncertainty, chi_squared       

# create the root window and an instance of Main Gui
if __name__ == "__main__":
    root = tk.Tk()
    app = MainGUI(root)
    root.mainloop()