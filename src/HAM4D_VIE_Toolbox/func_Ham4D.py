# Import packages
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------
# Functions used to manipulate and clean HAM4D_VIE-Data
# ---------------------------------------------------------

# Function calculate the mean-deviation of two columns
def deltaMeanPercentage(columnSim, columnSensor):
    """
    :param columnSim: column of dataframe with simulation data
    :param columnSensor: column of dataframe with sensor data
    """
    dF = (columnSim - columnSensor) / columnSensor * 100
    return dF.abs().mean()


# Function to clean recFilm-File from HAM4D_VIE
def recFilmTxtCleaner(filePath):
    filePathNew = filePath + '_Pandas.txt'
    f = open(filePath, 'r')
    fileLines = f.readlines()
    del fileLines[0:3]
    f.close()
    f = open(filePathNew, 'w')
    f.writelines(fileLines)
    f.close()
    return filePathNew


def recFilmTxtCleaner_Nodes(filePath):
    filePathNew = filePath + '_Pandas.txt'
    f = open(filePath, 'r')
    fileLines = f.readlines()
    del fileLines[0]
    f.close()
    f = open(filePathNew, 'w')
    f.writelines(fileLines)
    f.close()
    return filePathNew


# Function to drop unnecessary columns from HAM4D_VIE with connections
def dropEmptyColumnsSimData_verb(dF_Sim, numberOfCells):
    """
    Function to drop the columns with only the column name or unncessary data from the recFilm-File
    """
    if numberOfCells > 1:
        dF_Sim = dF_Sim.drop(
            columns=[1, 3 + (numberOfCells - 1), 5 + (numberOfCells - 1) * 2,
                     7 + (numberOfCells - 1) * 3, 9 + (numberOfCells - 1) * 4, 11 + (numberOfCells - 1) * 5,
                     13 + (numberOfCells - 1) * 6, 15 + (numberOfCells - 1) * 7,
                     17 + (numberOfCells - 1) * 8, 19 + (numberOfCells - 1) * 9,
                     21 + (numberOfCells - 1) * 10])
        # hinteren Informationszellen von Verbindungen droppen (ad.: Spalten nun anders nummeriert, weil
        # schon ein drop stattgefunden hat)
        dF_Sim = dF_Sim.drop(columns=dF_Sim.columns[23 - 11 + (numberOfCells - 1) * 11:])
    else:
        dF_Sim = dF_Sim.drop(columns=[1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 24, 25])
    return dF_Sim


# Funciton to name columns from HAM4D_VIE with connections
def name_col_dataframe_verb(dataframe, numberOfCells, listNameOfCells):
    """
    :param property: name of the property that will be passed as a string
    :param numberOfCells: numberOfCells
    :param dataframe: Dataframe where columns are to name
    :param listNameOfCells: List with the names and or materials of the cells
    :return: dataframe with named columns
    """
    listNameOfVerb = list(map(lambda sub: int(''.join(
        [ele for ele in sub if ele.isnumeric()])), listNameOfCells))

    # if property == 'temp':
    #     loopFactor = 0
    # elif property == 'relhum':
    #     loopFactor = 1
    # elif property == 'wat':
    #     loopFactor = 2
    # elif property == 'mould':
    #     loopFactor = 3

    count = 0
    for i in range(0 * numberOfCells + 1, (0 + 1) * numberOfCells + 1):
        dataframe.rename(columns={dataframe.columns[i]: 'temp_' + listNameOfCells[count]}, inplace=True)
        count += 1
    count = 0
    for i in range(1 * numberOfCells + 1, (1 + 1) * numberOfCells + 1):
        dataframe.rename(columns={dataframe.columns[i]: 'relhum_' + listNameOfCells[count]}, inplace=True)
        count += 1
    count = 0
    for i in range(2 * numberOfCells + 1, (2 + 1) * numberOfCells + 1):
        dataframe.rename(columns={dataframe.columns[i]: 'wat_' + listNameOfCells[count]}, inplace=True)
        count += 1
    count = 0
    for i in range(3 * numberOfCells + 1, (3 + 1) * numberOfCells + 1):
        dataframe.rename(columns={dataframe.columns[i]: 'mould_' + listNameOfCells[count]}, inplace=True)
        count += 1
    count = 0
    for i in range(4 * numberOfCells + 1, (4 + 1) * numberOfCells + 1):
        dataframe.rename(columns={dataframe.columns[i]: 'mould_verb_AO_' + str(listNameOfVerb[count])}, inplace=True)
        count += 1
    count = 0
    for i in range(5 * numberOfCells + 1, (5 + 1) * numberOfCells + 1):
        dataframe.rename(columns={dataframe.columns[i]: 'mould_verb_OB_' + str(listNameOfVerb[count])}, inplace=True)
        count += 1
    count = 0
    for i in range(6 * numberOfCells + 1, (6 + 1) * numberOfCells + 1):
        dataframe.rename(columns={dataframe.columns[i]: 'tempO_' + str(listNameOfVerb[count])}, inplace=True)
        count += 1
    count = 0
    for i in range(7 * numberOfCells + 1, (7 + 1) * numberOfCells + 1):
        dataframe.rename(columns={dataframe.columns[i]: 'vapourpressAO_' + str(listNameOfVerb[count])}, inplace=True)
        count += 1
    count = 0
    for i in range(8 * numberOfCells + 1, (8 + 1) * numberOfCells + 1):
        dataframe.rename(columns={dataframe.columns[i]: 'vapourpressOB_' + str(listNameOfVerb[count])}, inplace=True)
        count += 1
    count = 0
    for i in range(9 * numberOfCells + 1, (9 + 1) * numberOfCells + 1):
        dataframe.rename(columns={dataframe.columns[i]: 'relhumAO_' + str(listNameOfVerb[count])}, inplace=True)
        count += 1
    count = 0
    for i in range(10 * numberOfCells + 1, (10 + 1) * numberOfCells + 1):
        dataframe.rename(columns={dataframe.columns[i]: 'relhumOB_' + str(listNameOfVerb[count])}, inplace=True)
        count += 1
    return dataframe


# Funciton for scaling the 'relhum'-columns time 100 for HAM4D_VIE with connections
def relhumScaler_verb(dF_Sim, numberOfCells, listNameOfCells):
    listNameOfVerb = list(map(lambda sub: int(''.join(
        [ele for ele in sub if ele.isnumeric()])), listNameOfCells))
    count = 0
    for i in range(1 * numberOfCells + 1, (1 + 1) * numberOfCells + 1):
        dF_Sim['relhum_' + listNameOfCells[count]] = dF_Sim['relhum_' + listNameOfCells[count]] * 100
        count += 1
    count = 0
    for i in range(9 * numberOfCells + 1, (9 + 1) * numberOfCells + 1):
        dF_Sim['relhumAO_' + str(listNameOfVerb[count])] = dF_Sim['relhumAO_' + str(listNameOfVerb[count])] * 100
        count += 1
    count = 0
    for i in range(10 * numberOfCells + 1, (10 + 1) * numberOfCells + 1):
        dF_Sim['relhumOB_' + str(listNameOfVerb[count])] = dF_Sim['relhumOB_' + str(listNameOfVerb[count])] * 100
        count += 1
    return dF_Sim


# Same functions but for the older HAM4D_VIE version

def dropEmptyColumnsSimData(dF_Sim, numberOfCells):
    """
    Function to drop the columns with only the column name from the recFilm-File
    """
    if numberOfCells > 1:
        dF_Sim = dF_Sim.drop(
            columns=[1, 3 + (numberOfCells - 1), 5 + (numberOfCells - 1) * 2,
                     7 + (numberOfCells - 1) * 3])
    else:
        dF_Sim = dF_Sim.drop(columns=[1, 3, 5, 7])
    return dF_Sim


def name_col_dataframe(dataframe, numberOfCells, listNameOfCells):
    """
    :param property: name of the property that will be passed as a string
    :param numberOfCells: numberOfCells
    :param dataframe: Dataframe where columns are to name
    :param listNameOfCells: List with the names and or materials of the cells
    :return: dataframe with named columns
    """
    # if property == 'temp':
    #     loopFactor = 0
    # elif property == 'relhum':
    #     loopFactor = 1
    # elif property == 'wat':
    #     loopFactor = 2
    # elif property == 'mould':
    #     loopFactor = 3

    count = 0
    for i in range(0 * numberOfCells + 1, (0 + 1) * numberOfCells + 1):
        dataframe.rename(columns={dataframe.columns[i]: 'temp_' + listNameOfCells[count]}, inplace=True)
        count += 1
    count = 0
    for i in range(1 * numberOfCells + 1, (1 + 1) * numberOfCells + 1):
        dataframe.rename(columns={dataframe.columns[i]: 'relhum_' + listNameOfCells[count]}, inplace=True)
        count += 1
    count = 0
    for i in range(2 * numberOfCells + 1, (2 + 1) * numberOfCells + 1):
        dataframe.rename(columns={dataframe.columns[i]: 'wat_' + listNameOfCells[count]}, inplace=True)
        count += 1
    count = 0
    for i in range(3 * numberOfCells + 1, (3 + 1) * numberOfCells + 1):
        dataframe.rename(columns={dataframe.columns[i]: 'mould_' + listNameOfCells[count]}, inplace=True)
        count += 1
    return dataframe


def relhumScaler(dF_Sim, numberOfCells, listNameOfCells):
    count = 0
    for i in range(1 * numberOfCells + 1, (1 + 1) * numberOfCells + 1):
        dF_Sim['relhum_' + listNameOfCells[count]] = dF_Sim['relhum_' + listNameOfCells[count]] * 100
        count += 1
    return dF_Sim


# Function to clean RB-File from HAM4D_VIE
def climateTxtCleaner(filePath):
    filePathNew = filePath + '_Pandas.txt'
    f = open(filePath, 'r')
    fileLines = f.readlines()
    del fileLines[0:3]
    f.close()
    f = open(filePathNew, 'w')
    f.writelines(fileLines)
    f.close()
    return filePathNew


# Function to create dataframe with results
def create_result_dF(filePath, listNameOfCells):
    '''
    Function to create dataframe with results based on the recFilm-File.
    :param filePath: The path to the recFilm File as a string.
    :param listNameOfCells: A list with the names for the cells you are creating the results for.
    :return: Dataframe with the results based on the recFilm-File.
    '''
    # Simulation-Data
    # reading in the data
    fPath = recFilmTxtCleaner(filePath=filePath)
    # Input for creating and cleaning the dataframe
    dF_Ham = pd.read_csv(fPath, header=None, delim_whitespace=True, engine='c')
    # Defining the number of Cells in the recFilm-file
    numberOfCells = len(listNameOfCells)
    # Creating a function-pipe to clean the data
    dF_Ham = (dF_Ham.pipe(dropEmptyColumnsSimData_verb, numberOfCells=numberOfCells).pipe(
        name_col_dataframe_verb, listNameOfCells=listNameOfCells, numberOfCells=numberOfCells).pipe(
        relhumScaler_verb, listNameOfCells=listNameOfCells, numberOfCells=numberOfCells))
    return dF_Ham


# --------------------------------------------------------------------------
#       Functions to calculate specific parameters
# -------------------------------------------------------------------------


# Function to calculate the specific air humidity when using dataframes
def specificAirHumidity(tempSeries, relhumSeries):
    # Datensatz erstellen
    dF_x = pd.DataFrame()
    dF_x['tempSeries'] = tempSeries
    dF_x['relhumSeries'] = relhumSeries

    # Prüfen ob relhum in Prozent ist, falls ja, dann umrechnen
    dF_x['relhumSeries'] = dF_x['relhumSeries'].apply(lambda x: x / 100 if x > 1 else x)

    # der Luftdruck wird mit 101325 Pa angenommen
    pAir = 101325

    # pSat temperaturabhängig berechnen
    dF_x['pSat'] = dF_x['tempSeries'].apply(
        lambda x: 610.5 * np.exp(17.269 * x / (237.3 + x)) if x >= 0 else 610.5 * np.exp(21.875 * x / (265.5 + x)))

    # Dampfdruck berechnen
    dF_x['pVap'] = dF_x['pSat'] * dF_x['relhumSeries']

    # spezifische Luftfeuchtigkeit berechnen
    x = 0.622 * (dF_x['pVap'] / pAir)
    return x


# Function to calculate the absolute air humidity when using dataframes
def absoluteAirHumidity(tempSeries, relhumSeries):
    # Datensatz erstellen
    dF_x = pd.DataFrame()
    dF_x['tempSeries'] = tempSeries
    dF_x['relhumSeries'] = relhumSeries

    # Prüfen ob relhum in Prozent ist, falls ja, dann umrechnen
    dF_x['relhumSeries'] = dF_x['relhumSeries'].apply(lambda x: x / 100 if x > 1 else x)

    # spezifische Luftfeuchtigkeit berechnen
    x = (6.112 * np.exp((17.67 * dF_x['tempSeries']) / (dF_x['tempSeries'] + 243.5)) * dF_x[
        'relhumSeries'] * 100 * 2.1674) / \
        (273.15 + dF_x['tempSeries']) * 1 / 1000
    return x

    # https://carnotcycle.wordpress.com/2012/08/04/how-to-convert-relative-humidity-to-absolute-humidity/


# Function to calculate vapourpressure from series
def vapourpressure(tempSeries, relhumSeries):
    # Datensatz erstellen
    dF_pVap = pd.DataFrame()
    dF_pVap['tempSeries'] = tempSeries
    dF_pVap['relhumSeries'] = relhumSeries

    # Prüfen ob relhum in Prozent ist, falls ja, dann umrechnen
    dF_pVap['relhumSeries'] = dF_pVap['relhumSeries'].apply(lambda x: x / 100 if x > 1 else x)

    # der Luftdruck wird mit 101325 Pa angenommen
    pAir = 101325

    # pSat temperaturabhängig berechnen
    dF_pVap['pSat'] = dF_pVap['tempSeries'].apply(
        lambda x: 610.5 * np.exp(17.269 * x / (237.3 + x)) if x >= 0 else 610.5 * np.exp(21.875 * x / (265.5 + x)))

    # Dampfdruck berechnen
    pVap = dF_pVap['pSat'] * dF_pVap['relhumSeries']
    return pVap


# Function to calculate pSat from series
def pSat(tempSeries):
    # Datensatz erstellen
    dF_pSat = pd.DataFrame()
    dF_pSat['tempSeries'] = tempSeries

    # pSat temperaturabhängig berechnen
    pSat = dF_pSat['tempSeries'].apply(
        lambda x: 610.5 * np.exp(17.269 * x / (237.3 + x)) if x >= 0 else 610.5 * np.exp(21.875 * x / (265.5 + x)))
    return pSat


# Function to calculate relhum from series
def relhum(tempSeries, pVapSeries):
    # Datensatz erstellen
    dF_pSat = pd.DataFrame()
    dF_pSat['tempSeries'] = tempSeries
    dF_pSat['pVapSeries'] = pVapSeries

    # pSat temperaturabhängig berechnen
    pSat = dF_pSat['tempSeries'].apply(
        lambda x: 610.5 * np.exp(17.269 * x / (237.3 + x)) if x >= 0 else 610.5 * np.exp(21.875 * x / (265.5 + x)))

    # relhum berechnen
    relhum = pVapSeries / pSat

    return relhum


def f_LFK(temp_x, start, end):
    '''
    Function to calculate the "Luftfeuchteklasse" based on EN ISO 13788:2012 dependant on the temperature
    :param temp_x: array with temperature
    :param start: constant in the beginning of Luftfeuchteklasse Diagram from -5 to 0
    :param end: constant in the end of Luftfeuchteklasse Diagram from 20 to 100
    :return:
    '''
    k = (end - start) / 20
    f_LFK = k * temp_x + start
    f_LFK = np.where(temp_x <= 0, start, f_LFK)
    f_LFK = np.where(temp_x >= 20, end, f_LFK)
    return f_LFK


def materialfeuchte_massen_prozent(rho, wat):
    '''

    This function does not scale the result. If you need percentage you may have to multiply with 100 in the end.

    :param rho: Density of the materialin kg/m³
    :param wat: value, column, list with moisture content in kg/m³
    :return: value or column with moisture content in relation to the density
    '''

    m_perc = wat / rho
    return m_perc


# -----------------------------------#
# Functions for Node Network Output  #
# -----------------------------------#

# Function to drop unnecessary columns from HAM4D_VIE with connections
def dropEmptyColumnsSimData_node(dF_Sim, numberOfNodes):
    """
    Function to drop the columns with only the column name or unncessary data from the recFilm-File
    """
    if numberOfNodes > 1:
        dF_Sim = dF_Sim.drop(
            columns=[0, 2 + (numberOfNodes - 1), 4 + (numberOfNodes - 1) * 2,
                     6 + (numberOfNodes - 1) * 3])
    else:
        dF_Sim = dF_Sim.drop(columns=[0, 2, 4, 6])
    return dF_Sim


# Function to name columns from HAM4D_VIE with connections
def name_col_dataframe_node(dataframe, numberOfNodes, listNameOfNodes):
    """
    :param property: name of the property that will be passed as a string
    :param numberOfNodes: numberOfCells
    :param dataframe: Dataframe where columns are to name
    :param listNameOfNodes: List with the names and or materials of the cells
    :return: dataframe with named columns
    """

    count = 0
    for i in range(0, numberOfNodes):
        dataframe.rename(columns={dataframe.columns[i]: 'temp_' + listNameOfNodes[count]}, inplace=True)
        count += 1
    count = 0
    for i in range(numberOfNodes, (2 * numberOfNodes)):
        dataframe.rename(columns={dataframe.columns[i]: 'relhum_' + listNameOfNodes[count]}, inplace=True)
        count += 1
    count = 0
    for i in range(numberOfNodes * 2, (3 * numberOfNodes)):
        dataframe.rename(columns={dataframe.columns[i]: 'airpressure_' + listNameOfNodes[count]}, inplace=True)
        count += 1

    count = 0
    for i in range(numberOfNodes * 3, (4 * numberOfNodes)):
        dataframe.rename(columns={dataframe.columns[i]: 'qAir_' + listNameOfNodes[count]}, inplace=True)
        count += 1

    return dataframe


# Function for scaling the 'relhum'-columns time 100 for HAM4D_VIE nodes
def relhumScaler_node(dF_Sim, numberOfNodes, listNameOfNodes):
    count = 0
    for i in range(numberOfNodes, 2 * numberOfNodes):
        dF_Sim['relhum_' + listNameOfNodes[count]] = dF_Sim['relhum_' + listNameOfNodes[count]] * 100
        count += 1

    return dF_Sim


# -------------------------
# Boundary Conditions
# ------------------------

def create_HAM4D_boundary_file(dataframe, filepath):
    '''
    Function to create HAM4D_VIE boundary files, based on a dataframe. This dataframe could be the result of a csv containing climate data for example.
    :param dataframe: Dataframe with the wanted boundary conditions, formatted as needed for HAM4D_VIE boundary files.
    :param filepath: filepath + filename to save the file to.
    :return:
    '''

    # Write the custom string in the beginning of the file
    custom_string = 'zeit(PeriodenEnde)_temp_alphaC_tempS_eps_rad_absG_phi_betaV_suc_qL_druck\n\n0	Referenzhöhe_f_Druck_in_m\n'

    with open(filepath, 'w') as file:
        file.write(custom_string + '\n')
        dataframe.to_csv(file, index=False, header=None, sep='\t')
        file.close()

    return


def insert_missing_data_with_reverse(dataframe_to_mirror, index):
    '''Function to mirror the data where no measured data is available'''
    dF_new = dataframe_to_mirror
    dF_reverse = dataframe_to_mirror.iloc[::-1].reset_index(drop=True)
    dF_new.iloc[index:] = dF_reverse.iloc[index:]
    return dF_new


def sin_climate_creater(a, b, time):
    '''
    Function to create arbitrary syntehtic climates.
    :param a: Lower threshold value
    :param b: Upper threshold value
    :param time: Timespan or x-values
    :return: Array with adapted sinewave based on input
    '''

    start_time = 0
    end_time = 1
    time = np.arange(start_time, end_time, 1 / time)
    theta = -1.9102
    sinewave = (a + b) / 2 + (b - a) / 2 * np.sin(2 * np.pi * time + theta)
    return sinewave


# -------------------------
# Risk Analysis
# ------------------------

# Calc Rot Wood
def rot_risk_wood(dF_temp_series, dF_rh_series, varname, savefig, **kwargs):
    '''
    Function to calculate the risk of rot based on "WTA-Merkblatt" based on the temperature and rh.
    :param dF_temp_series: dF series of mean daily temperature
    :param dF_rh_series: dF series of mean daily rh
    :param savefig: bool to save fig.
    :param plot_dir: path where to save as string
    :param filename: name of file
    :return: shows plot and saves figure
    '''


    # Calculate threshholds for plots
    coord_start_rot = np.array([0, 30])
    coord_end_rot = np.array([95, 86])

    plt.plot(dF_temp_series[0:365], dF_rh_series[0:365],
             label=f'Ergebnisse_{varname}_1._Jahr', marker='o', linestyle='')
    plt.plot(dF_temp_series[1460:1825], dF_rh_series[1460:1825],
             label=f'Ergebnisse_{varname}_5._Jahr', marker='o', linestyle='')
    plt.plot(dF_temp_series[3285:3650], dF_rh_series[3285:3650],
             label=f'Ergebnisse_{varname}_10._Jahr', marker='o', linestyle='')
    plt.plot(coord_start_rot, coord_end_rot, label='Grenzwert Holzverrottung', color='red')
    plt.legend()
    plt.xlabel('Temperatur in Grad Celsius')
    plt.ylabel('rel. Porenluftfeuchtigkeit in %')
    plt.xlim(0, 30)
    plt.grid()
    plt.tight_layout()
    if savefig == True:
        plot_dir = kwargs.get('plot_dir', None)
        plt.savefig(plot_dir + f'/{varname}_rot_risk.pdf')


# Cals Structural Integrity Wood

def struct_risk_wood(dF_wat_series, savefig, varname, **kwargs):
    '''
    Function to calculate the risk of structural damage and integrity in wood based on "WTA-Merklblatt".
    :param dF_temp_series: dF series of mean daily wat content in kg/m³
    :param savefig: bool to save fig.
    :param plot_dir: path where to save as string
    :param filename: name of file
    :return: shows plot and saves figure
    '''
    m_perc_Sparren = materialfeuchte_massen_prozent(450, dF_wat_series) * 100
    #%%
    plt.plot(m_perc_Sparren, label=f'M%_Sparren_{varname}')
    plt.axhline(18, xmin=0.025, xmax=1, label='M%_Grenzwert', color='orange')
    plt.axhline(20, xmin=0, xmax=0.025, label='M%_Grenzwert_ersten_3_Monate', color='red')

    plt.legend()
    plt.xlabel('Zeit in Tagen')
    plt.ylabel('Materialfeuchte in M-%')
    plt.xlim(0, 3650)
    plt.grid()
    plt.tight_layout()
    if savefig == True:
        plot_dir = kwargs.get('plot_dir', None)
        plt.savefig(plot_dir + f'/{varname}_struct_risk.pdf')