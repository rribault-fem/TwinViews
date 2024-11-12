from setuptools import setup, find_packages

setup(
    name='TwinViews',
    version='0.1.0',
    url='https://gitlab.france-energies-marines.org/Romain/saitec_demosath',
    author='Romain Ribault',
    author_email='romain.ribault@france-energies-marines.org',
    description='Explore Demosath simulation and measurements data',
    packages=find_packages(),    
    install_requires=['numpy==2.0.0', 'pandas[performance, plot, excel]==2.2.2', 'xarray[complete]==2024.5.0',
                    'plotly==5.22.0', 'shiny==0.10.2', 'shinywidgets==0.3.2', 'scipy', 'seaborn', 'pydantic==2.8.2'],
)