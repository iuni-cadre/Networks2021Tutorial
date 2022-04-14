# Networks 2021 Tutorial and IUNI 2022 Workshop

Repository with scripts (Jupyter notebooks) and demo data for the CADRE Networks 2021 Tutorial and IUNI 2022 Workshop: Conducting Science of Science Research on the CADRE Platform. 

For more information on the CADRE project, check our website: https://cadre.iu.edu

Also check *CADRE in 5: #1 What is CADRE?*
https://www.youtube.com/watch?v=IlgpjOvkojQ&t=2s

### Usage inside the CADRE platform
The examples are designed to run in the CADRE platform jupyter enviroment. First, log in to CADRE website.
To open the Jupyter environment, click on the button `Jupyter Notebook`, then `Start Notebook Server` and wait for it to open. Note that your browser may block the popup window, so click in the icon corresponding to the popup windows in your browser to allow it for the CADRE website.

 - If your account is new, the files of this workshop will already be available to you in a folder named `2021Tutorial`. The files are now ready to be used in the platform.

 - If your account is older, the files of this workshop may need to be updated. Inside the Jupyter environment, click on menu `File`>`New`>`Terminal` and type in the terminal:
   ```bash
   git clone https://github.com/iuni-cadre/Networks2021Tutorial.git Workshop2022
   ```
   The files are now available under the `Workshop2022` folder.

You can now run the examples using the demo data or you can use the Query Builder to generate your own queries. Note that examples 2-4 require the option "Include Citation Network Graph" to be enabled in the query builder, so that the citation networks can be constructed.

#### Using your own query results
In all the examples, a variable named `queryID` can be set to use data from any query results you have under your account. You can check the available results in the `query-results` folder. Set queryID to a composition of `query name`+`_`+`query identifier` , such as in the filenames in the `query-results` folder without the `.csv` or `_edges.csv` suffixes.


### Testing the demos locally in your machine
These demos can also be tested localy using the demo queries.
First, a `python >= 3.8` environment is required. If you use conda (https://docs.conda.io/en/latest/miniconda.html), we recommend to create and activate a new environment by using:

```bash
conda create --name=cadreworkshop -c conda-forge python=3.8
conda activate cadreworkshop
```
Then download the project by using the github web interface or via git:
```bash
git clone https://github.com/iuni-cadre/Networks2021Tutorial.git Workshop2022
```
Dont forget to go the new folder by using `cd Workshop2022`.

Next, the required packages should be installed using pip:
```bash
pip install -r requirements.txt
```

Finally, the environment can be started by initializing the jupyter lab environment:
```bash
jupyter lab
```

This should open the jupyter notebook in your browser.


#### Run the tutorials online via Binder
Binder can also be used to run the tutorials. However, only the demo queries can be used. Click on the link below to start a free binder environment:
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/iuni-cadre/Networks2021Tutorial/HEAD)
