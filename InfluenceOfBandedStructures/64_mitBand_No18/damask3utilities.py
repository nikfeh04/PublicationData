"""
10.11.2021
Author: Niklas Fehlemann

Different utility functions to handle damask 3 Output
"""
import os
import damask
import sys
import copy
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import pyvista as pv
import matplotlib


def add_strains(fname):
    """
    Adds the strains and stresses to a hdf5-outputfile
    WARNING Modifies file in-place and can only be called once!
    """
    print('Adding Mises')
    try:
        result = damask.Result(fname)
        result.add_stress_Cauchy()
        result.add_strain()
        result.add_equivalent_Mises('sigma')
        result.add_equivalent_Mises('epsilon_V^0.0(F)')
    except:
        print('DSet already there')


def calc_strain_partitioning(fname, output_path, save=True):
    """
    Calcs the flowcurve-partitioning (Flowcurves for Ferrite and Martensite separated)
    params:
        - fname: Damask-hdf5 with stresses and strains (as String)
        - output_path: Path to save
    """
    result = damask.Result(fname)
    try:
        result.get('sigma_vM')
    except:
        print('No MisesStress in DSet!')
        sys.exit(0)

    # Stress - strain Partioning
    stresses_f = list()
    strains_f = list()
    stresses_m = list()
    strains_m = list()
    for keys, values in result.get('sigma_vM').items():
        ferrite = values['Ferrite']
        martensite = values['Martensite']
        stresses_f.append(ferrite.mean())
        stresses_m.append(martensite.mean())

    for keys, values in result.get('epsilon_V^0.0(F)_vM').items():
        ferrite = values['Ferrite']
        martensite = values['Martensite']
        strains_f.append(ferrite.mean())
        strains_m.append(martensite.mean())

    plt.plot(strains_f, stresses_f)
    plt.plot(strains_m, stresses_m)
    plt.xlabel('True Strain')
    plt.ylabel('True Stress')
    plt.legend(['Ferrit', 'Martensite'])
    if save:
        plt.savefig(output_path + '/FlowcurvePartitioning.png')
    plt.close()

    return [stresses_f, strains_f], [stresses_m, strains_m]


def calc_comparison(fname, output_path, hollomon_voce=None):
    """
    Calcs the flowcurve and the comparison between real and RVE-Flowcurve
    WARNING: Currently only supports hollomon-voce fits
    params:
        - fname: Damask-hdf5 file (as String)
        - output_path: Path to save
        - hollomon_voce: parameters for hollomon-voce fit (list)
    """
    if hollomon_voce is None:
        hv = [0.5138, 1843, 0.44, 1167, 820.4, 1167, -100]
    else:
        hv = hollomon_voce
    result = damask.Result(fname)
    try:
        result.get('sigma_vM')
    except:
        print('No MisesStress in DSet!')
        sys.exit(0)

    number_of_phases = 2  # Optionen: 1 für Einphasige RVEs, 2 für zweiphasige RVEs

    strain_eq = result.get(
        'epsilon_V^0.0(F)_vM')  # Vergleichsdehnung aus dem Result herausziehen. Für alle Incremente und Zellen
    stress_eq = result.get('sigma_vM')  # Das gleiche für die Spannung
    stress_vector = []
    strain_vector = []
    if number_of_phases == 1:  # Für einphasige RVEs
        for key, value in strain_eq.items():  # Iteriere über Incremente
            strain_vector.append(value.mean())  # Bilde Mitterlwert über das Gitter

        for key, value in stress_eq.items():
            stress_vector.append(value.mean())

    elif number_of_phases == 2:  # Für zweiphasige RVEs

        for _, tuple_strain in strain_eq.items():  # Iteriere über Incremente
            [phase1, phase2] = tuple_strain.items()  # Teile die Phasen auf
            [_, strain_phase1] = phase1  # isoliere die Dehnung an den einzelnen Gitterpunkten
            [_, strain_phase2] = phase2
            total_strain = np.append(strain_phase1, strain_phase2)  # Hänge beide Vectoren hintereinander
            strain_vector.append(
                total_strain.mean())  # Bilde Mittelwert über das Gitter und füge es der Liste hinzu

        for _, tuple_stress in stress_eq.items():
            [phase1, phase2] = tuple_stress.items()
            [_, stress_phase1] = phase1
            [_, stress_phase2] = phase2
            total_stress = np.append(stress_phase1, stress_phase2)
            stress_vector.append(total_stress.mean())

    del strain_vector[0]  # Elastic
    del stress_vector[0]

    stress_vector = np.asarray(stress_vector) / 1000000

    # Calc the hollomon-Voce stress
    def hollomonVoce(strain):
        hv_stress = hv[0] * (hv[1] * np.asarray(strain) ** hv[2]) + (1 - hv[0]) * (
                hv[3] + (hv[4] - hv[5]) * np.exp(hv[6] * np.asarray(strain)))
        return hv_stress

    real_stress = hollomonVoce(strain_vector)

    # Return data and calc loss
    loss = np.sqrt(mean_squared_error(stress_vector, real_stress))  # RMSE instead of MSE

    # Plotting
    plt.figure(figsize=(15, 10))
    plt.plot(strain_vector, stress_vector)
    plt.savefig(output_path + '/FlowcurveRVE.png')
    plt.plot(strain_vector, real_stress)
    plt.title('Loss is {}'.format(loss))
    plt.legend(['RVE', 'HollomonVoceRealFit'])
    plt.savefig(output_path + '/FlowcurveComparison.png')
    plt.close()


def calc_barplots(fname, output_path, increment, kind='strain', save=True, separate_band=False, type='barplot'):
    """
    Makes some barplots for given increments and strain or stress
    params:
        - fname: (string) path to hdf5 file
        - output_path: (string) where to save the plots
        - increment: (int or list of ints) increments to plot
        - kind: (string) stress or strain to plot: default: strain
    """
    result = damask.Result(fname)
    try:
        result.get('sigma_vM')
    except:
        print('No MisesStress in DSet!')
        sys.exit(0)
    pass

    data = result.view('increments', increment)

    if kind.lower() == 'strain':
        identifier = 'epsilon_V^0.0(F)_vM'
    elif kind.lower() == 'stress':
        identifier = 'sigma_vM'
    else:
        print('no valid identifier! has to be "strain" or "stress"')
        sys.exit(0)

    df1 = pd.DataFrame(data.get(identifier)['Martensite'], columns=[kind])
    df1['Type'] = 'Martensite'

    df2 = pd.DataFrame(data.get(identifier)['Ferrite'], columns=[kind])
    df2['Type'] = 'Ferrite'

    if not separate_band:
        df = pd.concat([df1, df2])

        ratio = df1[kind].to_numpy().mean() / df2[kind].to_numpy().mean()
        if type == 'violinplot':
            sns.violinplot(data=df, y=kind, x='Type')
        elif type == 'barplot':
            sns.barplot(data=df, y=kind, x='Type')
        elif type == 'boxplot':
            sns.boxplot(data=df, y=kind, x='Type', showfliers=False)
        plt.title('{} - OverallRatio is: {}'.format(kind, ratio))
        if save:
            plt.savefig(output_path + '/Distributions_{}_inc{}_{}_{}'.format(kind, increment, separate_band, type))
        plt.close()

        return df1, df2

    elif separate_band:
        input_grid = damask.Grid.load(output_path + '/grid.vti')
        band_idx = input_grid.material.max()
        points = input_grid.material.flatten(order='F') == band_idx

        data = result.view('increments', increment).place(identifier)

        banded_martensite = data[points]

        df3 = pd.DataFrame(banded_martensite, columns=[kind])
        df3['Type'] = 'BandedMartensite'
        df = pd.concat([df2, df3])

        ratio = df3[kind].to_numpy().mean() / df2[kind].to_numpy().mean()
        if type == 'violinplot':
            sns.violinplot(data=df, y=kind, x='Type')
        elif type == 'barplot':
            sns.barplot(data=df, y=kind, x='Type')
        elif type == 'boxplot':
            sns.boxplot(data=df, y=kind, x='Type', showfliers=False)
        plt.title('{} - BandRatio is: {}'.format(kind, ratio))
        if save:
            plt.savefig(output_path + '/Distributions_{}_inc{}_{}_{}'.format(kind, increment, separate_band, type))
        plt.close()

        return df3, df2


def calc_volume(fname) -> dict:
    """
    Calculates the "volume" of a RVE using the points of the different phases
    """
    result = damask.Result(fname)

    data = result.view('increments', 0)

    npts_ferrite = data.get('sigma_vM')['Ferrite'].__len__()
    npts_martensite = data.get('sigma_vM')['Martensite'].__len__()

    percentage_ferrite = npts_ferrite / (npts_ferrite + npts_martensite)
    percentage_martensite = npts_martensite / (npts_ferrite + npts_martensite)

    vols = {'Ferrite': percentage_ferrite, 'Martensite': percentage_martensite}

    return vols


def aggregate_flowcurves(dir):
    """
    Use data from strain partitioning to aggreate flowcurves for one folder
    Plot the corresponding volume also
    """
    path_list = os.listdir(dir)
    liste = list()

    plt.figure(figsize=(16, 12))
    for i in range(path_list.__len__()):
        try:
            _, martensite = calc_strain_partitioning(dir + path_list[i] + '/grid_load.hdf5',
                                                     output_path=dir + path_list[i],
                                                     save=False)
            liste.append(martensite)
        except:
            pass

    for i in range(path_list.__len__()):
        try:
            if 'mit' in path_list[i]:
                c = 'blue'
            else:
                c = 'orange'
            plt.plot(liste[i][1], liste[i][0], color=c, label=path_list[i])
        except:
            pass

    plt.legend()

    if not os.path.isdir(dir + '/PostProc'):
        os.mkdir(dir + '/PostProc')

    plt.savefig(dir + '/PostProc' + '/FlowcurveAggregates.png')


def aggregate_boxplot(dir, increment, kind, partitioned=False, separate_band=False, type='boxplot'):
    """
    Use data from strain partitioning to aggreate a boxplot for one folder
    """
    path_list = os.listdir(dir)
    martensite_df = pd.DataFrame()

    for i in range(path_list.__len__()):
        print(dir + path_list[i] + '/grid_load.hdf5')
        try:
            if 'mit' in path_list[i]:
                martensite, _ = calc_barplots(fname=dir + path_list[i] + '/grid_load.hdf5',
                                              kind=kind, increment=increment, save=False,
                                              output_path=path_list[i], separate_band=separate_band)
                martensite['Label'] = path_list[i]
                martensite['Type'] = 'Banded'
            else:
                martensite, _ = calc_barplots(fname=dir + path_list[i] + '/grid_load.hdf5',
                                              kind=kind, increment=increment, save=False,
                                              output_path=path_list[i], separate_band=False)
                martensite['Label'] = path_list[i]
                martensite['Type'] = 'Not Banded'
            martensite_df = pd.concat([martensite_df, martensite])
        except:
            pass

    """Calc the mean values"""
    mean_band = float(martensite_df[martensite_df['Type'] == 'Banded'][kind].mean())
    mean_unband = float(martensite_df[martensite_df['Type'] == 'Not Banded'][kind].mean())
    print(mean_band)
    print(mean_unband)
    if not partitioned:
        fig = plt.figure(figsize=(16, 12))
        ax = fig.add_subplot(111)
        if type == 'violinplot':
            sns.violinplot(data=martensite_df, y=kind, x='Type', ax=ax)
        elif type == 'barplot':
            sns.barplot(data=martensite_df, y=kind, x='Type', ax=ax)
        elif type == 'boxplot':
            sns.boxplot(data=martensite_df, y=kind, x='Type', showfliers=False, ax=ax)
    elif partitioned:
        fig = plt.figure(figsize=(24, 12))
        ax = fig.add_subplot(111)
        if type == 'violinplot':
            sns.violinplot(data=martensite_df, y=kind, x='Type', ax=ax, hue='Label')
        elif type == 'barplot':
            sns.barplot(data=martensite_df, y=kind, x='Type', ax=ax, hue='Label')
        elif type == 'boxplot':
            sns.boxplot(data=martensite_df, y=kind, x='Type', showfliers=False, ax=ax, hue='Label')

    ax.yaxis.label.set_size(24)
    ax.xaxis.label.set_size(24)
    ax.tick_params(direction='out', length=6, width=2, colors='black',
                   grid_color='black', grid_alpha=0.5, labelsize=20)

    if kind == 'strain':
        ax.set_title('Banded {:.2f} - Unbanded {:.2f}'.format(mean_band * 100, mean_unband * 100), fontsize=28)
    else:
        ax.set_title(
            'Banded {:.2f} - Unbanded {:.2f}'.format(round(mean_band / 1000000, 2), round(mean_unband / 1000000, 2)),
            fontsize=28)

    if not os.path.isdir(dir + '/PostProc'):
        os.mkdir(dir + '/PostProc')

    type_martensite = 'onlyBands' if separate_band else 'OverallMartensite'
    fig.savefig(dir + '/PostProc' + '/{}_{}_{}_{}_{}.png'.format(type, kind, increment, partitioned,
                                                                 type_martensite))




def find_damage_initiation(fname, stress, nelem):
    """
    Loops over a hdf5-file and finds the earliest increment where:
        nelem >= stress
    params:
        - fname: path to hdf5 file
        - stress: damage initiation stress
        - nelem: Number of elements to count as damage initiation site
    """


"""
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++
PyVista Stuff!
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++
"""


def export_custom_vtk(fname, grid, output_path, increment, phase='all') -> str:
    """
    Exports vtk to given folder
    params:
        fname: path to hdf5
        grid: path to vti-input
        output_path: where to save the vtk
        increment: The increment to view
        phase: 'all', 'Ferrite', 'Martensite'
    """
    result = damask.Result(fname=fname)
    result = result.view('increment', increment)
    grid = damask.Grid.load(grid)

    if phase.lower() == 'all':
        pass
    elif phase.lower() == 'ferrite':
        result = result.view('phase', 'Ferrite')
    elif phase.lower() == 'martensite':
        result = result.view('phase', 'Martensite')

    cwd = os.getcwd()
    os.chdir(output_path)
    result.export_VTK(mode='cell', parallel=False, fill_float=0.0, fill_int=0)
    if increment < 10:
        os.rename('grid_load_inc00{}.vti'.format(increment),
                  'grid_load_inc00{}_{}.vti'.format(increment, phase.lower()))
        result_grid = pv.read('grid_load_inc00{}_{}.vti'.format(increment, phase.lower()))
        os.remove('grid_load_inc00{}_{}.vti'.format(increment, phase.lower()))
        result_grid.cell_arrays['grain'] = grid.material.flatten(order='F')
        result_grid.save('grid_load_inc00{}_{}.vti'.format(increment, phase.lower()))
    elif 10 < increment < 100:
        os.rename('grid_load_inc0{}.vti'.format(increment), 'grid_load_inc0{}_{}.vti'.format(increment, phase.lower()))
        result_grid = pv.read('grid_load_inc0{}_{}.vti'.format(increment, phase.lower()))
        os.remove('grid_load_inc0{}_{}.vti'.format(increment, phase.lower()))
        result_grid.cell_arrays['grain'] = grid.material.flatten(order='F')
        result_grid.save('grid_load_inc0{}_{}.vti'.format(increment, phase.lower()))
    else:
        os.rename('grid_load_inc{}.vti'.format(increment), 'grid_load_inc{}_{}.vti'.format(increment, phase.lower()))
        result_grid = pv.read('grid_load_inc{}_{}.vti'.format(increment, phase.lower()))
        os.remove('grid_load_inc{}_{}.vti'.format(increment, phase.lower()))
        result_grid.cell_arrays['grain'] = grid.material.flatten(order='F')
        result_grid.save('grid_load_inc{}_{}.vti'.format(increment, phase.lower()))
    os.chdir(cwd)

    if increment < 10:
        path_to_vti = output_path + '/grid_load_inc00{}_{}.vti'.format(increment, phase.lower())
    elif 10 < increment < 100:
        path_to_vti = output_path + '/grid_load_inc0{}_{}.vti'.format(increment, phase.lower())
    else:
        path_to_vti = output_path + '/grid_load_inc{}_{}.vti'.format(increment, phase.lower())

    return path_to_vti


def show_interactive_mises(fname, what='stress', separator='strain', kind='ge', thresh=1, clim=None):
    pv.set_plot_theme("document")
    mesh = pv.read(fname)  # lade vtk file und transformiere es in ein Uniform grid
    new_pos = mesh.points + mesh.get_array('u')  # add the disposition of the grid points to the original position
    mesh = mesh.cast_to_structured_grid()  # change it from uniform grid to structured grid to be able to edit position
    mesh.points = new_pos  # change the position of the gridpoints
    mesh = pv.wrap(mesh)  # wrap it up for visualization

    if separator == 'stress':
        separator = 'phase/mechanical/sigma_vM / Pa'
    elif separator == 'strain':
        separator = 'phase/mechanical/epsilon_V^0.0(F)_vM / 1'

    if kind == 'equal':
        mask = np.where((mesh.cell_arrays[separator] == thresh))
    elif kind == 'le':
        mask = np.where((mesh.cell_arrays[separator] <= thresh))
    elif kind == 'ge':
        mask = np.where((mesh.cell_arrays[separator] >= thresh))
    else:
        raise Exception

    cell_ind = np.asarray(mask)
    mesh = mesh.extract_cells(cell_ind)

    if what == 'stress':
        ident = 'phase/mechanical/sigma_vM / Pa'
    elif what == 'strain':
        ident = 'phase/mechanical/epsilon_V^0.0(F)_vM / 1'

    if clim is None:
        max = mesh.cell_arrays[ident].max()
        min = mesh.cell_arrays[ident].min()
    else:
        min = clim[0]
        max = clim[1]

    plotter = pv.Plotter()
    plotter.add_text('{}'.format(what))  # Füge Name zum Subplot hinzu
    plotter.add_mesh_clip_plane(mesh, scalars=ident, show_edges=True, cmap='jet', clim=[min, max])
    plotter.show()


def make_screenshot(fname, output, cpos, what='stress', separator='strain', kind='ge', thresh=1, clim=None):
    pv.set_plot_theme("document")
    mesh = pv.read(fname)  # lade vtk file und transformiere es in ein Uniform grid
    new_pos = mesh.points + mesh.get_array('u')  # add the disposition of the grid points to the original position
    mesh = mesh.cast_to_structured_grid()  # change it from uniform grid to structured grid to be able to edit position
    mesh.points = new_pos  # change the position of the gridpoints
    mesh = pv.wrap(mesh)  # wrap it up for visualization

    if separator == 'stress':
        separator2 = 'phase/mechanical/sigma_vM / Pa'
    elif separator == 'strain':
        separator2 = 'phase/mechanical/epsilon_V^0.0(F)_vM / 1'

    if kind == 'equal':
        mask = np.where((mesh.cell_arrays[separator2] == thresh))
    elif kind == 'le':
        mask = np.where((mesh.cell_arrays[separator2] <= thresh))
    elif kind == 'ge':
        mask = np.where((mesh.cell_arrays[separator2] >= thresh))
    else:
        raise Exception

    cell_ind = np.asarray(mask)
    mesh = mesh.extract_cells(cell_ind)
    print(mesh)

    if what == 'stress':
        ident = 'phase/mechanical/sigma_vM / Pa'
    elif what == 'strain':
        ident = 'phase/mechanical/epsilon_V^0.0(F)_vM / 1'

    if clim is None:
        max = mesh.cell_arrays[ident].max()
        min = mesh.cell_arrays[ident].min()
    else:
        min = clim[0]
        max = clim[1]

    plotter = pv.Plotter()
    plotter.add_text('{}_{}_{}'.format(what, separator, thresh))  # Füge Name zum Subplot hinzu
    plotter.add_mesh(mesh, scalars=ident, show_edges=True, cmap='jet', clim=[min, max])
    plotter.show(screenshot=output + r'/Mises_{}_{}_{}_{}_{}.png'.format(what, separator, thresh, cpos, clim),
                 auto_close=True, cpos=cpos)


if __name__ == '__main__':
    output = os.getcwd()
    fname = output + '/grid_load.hdf5'
    gname = output + '/grid.vti'

    add_strains(fname)

    calc_comparison(fname, output_path=output)

    strains, stresses = calc_strain_partitioning(fname, output_path=output)

    calc_barplots(fname, kind='strain', increment=100, output_path=output)

    vols = calc_volume(fname)

    for i in [12, 16, 20, 24, 32, 40, 100]:
        export_custom_vtk(fname, output_path=output, grid=gname, phase='all', increment=i)

    with open(output + '/Specs.txt', 'a') as specs:
        specs.writelines('\n\n The grid volumes are: \n')
        for key, value in vols.items():
            specs.writelines('{} - {}\n'.format(key, round(value * 100, 5)))
