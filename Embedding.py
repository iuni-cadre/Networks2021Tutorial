#!/usr/bin/env python
# coding: utf-8

# ### Tutorial CADRE 2021
# 
# In this tutorial we will illustrate a modern Machine Learning pipeline applied to scholarly data. This should cover:
# 
#  * Loading query results data
#  * Building a citation network
#  * Obtaining an embedding of the network by employing Node2Vec
#    * Generating random walks sentences
#    * Train a Word2Vec model from it
#  * Generate 2D positions from the embedding by using UMAP
#  * Interactive visualization of the embedding
#  * SemAxis exploration

from tqdm.auto import tqdm
import pandas as pd
import xnetwork as xnet
import numpy as np
import igraph as ig
import sys
import os
import cxrandomwalk as rw
import umap
import gensim
import importlib
import matplotlib
matplotlib.use('Agg')
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from collections import OrderedDict
import matplotlib.patheffects as pe
import scipy.stats as scistats
# Semaxis support from the emlens package (https://github.com/skojaku/emlens)
# https://github.com/skojaku/emlens/blob/main/emlens/semaxis.py
import emlens_semaxis as emlens
from gensim.models.callbacks import CallbackAny2Vec


# Parameters
maxSamples = 20;  # max samples per class
nameAttribute = "Paper_originalTitle"
groupAttribute = "year"

groups = None

# You can define your own group rules here:

# groupAttribute ="JournalFixed_displayName"

# groups = {
#     "Nature Cell Biology": lambda v : v == "Nature Cell Biology",
#     "Nature Physics": lambda v : v == "Nature Physics",
# }




interactive = False

MAGColumnTypes = {
    "Paper_paperId":int,
    'Affiliation_displayName': str,
    'Author_authorId': str,
    'Author_rank': str,
    'Author_displayName': str,
    'Author_lastKnownAffiliationId': str,
    'Paper_bookTitle': str,
    'ConferenceInstance_conferenceInstanceId': str,
    'Paper_date': str,
    'Paper_docType': str,
    'Paper_doi': str,
    'FieldOfStudy_fieldOfStudyId': str,
    'Paper_issue': str,
    'JournalFixed_displayName': str,
    'JournalFixed_journalIdFixed': str,
    'JournalFixed_issn': str,
    'JournalFixed_publisher': str,
    'Paper_originalTitle': str,
    'Paper_citationCount': int,
    'Paper_estimatedCitation': int,
    'Paper_firstPage': str,
    'Paper_lastPage': str,
    'Paper_publisher': str,
    'Paper_referenceCount': int,
    'Paper_paperTitle': str,
    'Paper_year': int,
    'isQueryPaper': str,
}


# ### Function to build a citation network from the data

# In[5]:


def mag_query_input_to_xnet(nodes_file, edges_file, output_file):
    global edgesData,nodesData,vertexAttributes,index2ID,graph
    edgesData = pd.read_csv(edges_file)
    nodesData = pd.read_csv(nodes_file, dtype=MAGColumnTypes)

    # Replacing NaN for empty string
    for key in MAGColumnTypes:
        if(key in nodesData):
            nodesData[key].fillna("",inplace=True)

    # Generating continous indices for papers
    index2ID  = nodesData["Paper_paperId"].tolist()
    ID2Index = {id:index for index, id in enumerate(index2ID)}


    # Hack to account for 2 degree capitalized "FROM"
    fromKey = "From (Citing)"

    toKey = "To (Cited)"
    
    # Converting edges from IDs to new indices
    # Invert edges so it means a citation between from to to
    edgesZip = zip(edgesData[fromKey].tolist(),edgesData[toKey].tolist())
    edgesList = [(ID2Index[toID],ID2Index[fromID]) for fromID,toID in edgesZip if fromID in ID2Index and toID in ID2Index]

    vertexAttributes = {key:nodesData[key].tolist() for key in nodesData if key in MAGColumnTypes}
    
    for key,data in vertexAttributes.items():
        if (isinstance(data[0],str)):
            vertexAttributes[key] = [sEntry if len(sEntry)>0 else "None" for sEntry in [entry.strip("[]") for entry in data]]
            
    graph = ig.Graph(
        n=len(index2ID),
        edges=edgesList,
        directed=True,
        vertex_attrs=vertexAttributes
    )

    verticesToDelete = np.where(np.array([value=="false" for value in graph.vs["isQueryPaper"]]))[0]
    graph.delete_vertices(verticesToDelete)
    graph.vs["KCore"] = graph.shell_index(mode="IN")
    graph.vs["In-Degree"] = graph.degree(mode="IN")
    graph.vs["Out-Degree"] = graph.degree(mode="OUT")

    if("Paper_year" in graph.vertex_attributes()):
        graph.vs["year"] = [int(year) for year in graph.vs["Paper_year"]]
    else:
        graph.vs["year"] = [int(s[0:4]) for s in graph.vs["date"]]
    
    giantComponent = graph.clusters(mode="WEAK").giant()
    giantCopy = giantComponent.copy()
    giantCopy.to_undirected()
    giantComponent.vs["Community"] = [str(c) for c in giantCopy.community_multilevel().membership]
    xnet.igraph2xnet(giantComponent, output_file)


# #### Auxiliary functions

# In[8]:



class MonitorCallback(CallbackAny2Vec):
    def __init__(self,pbar):
        self.pbar = pbar

    def on_epoch_end(self, model):
        self.pbar.update(1)
        self.pbar.refresh() # to show immediately the update


def visualize_figures(networkPath,sentencesPath, modelPath,
                      exportFilename, export_file_semaxis,
                      export_file_correlation):
    g = xnet.xnet2igraph(networkPath)
    
    def make_pbar():
        pbar = None
        def inner(current,total):
            nonlocal pbar
            if(pbar is None):
                pbar= tqdm(total=total);
            pbar.update(current - pbar.n)
        return inner
    
    agent = rw.Agent(g.vcount(), np.array(g.get_edgelist()), False)
    agent.generateWalks(q=1.0, p=1.0, filename=sentencesPath, walksPerNode=10, verbose=False, updateInterval=100,
                        callback=make_pbar())  # filename="entireData.txt",
    monitor = MonitorCallback(tqdm())
    gensimModel = gensim.models.Word2Vec(
        gensim.models.word2vec.LineSentence(sentencesPath),
        vector_size=64,
        workers=2,
        min_count=1,
        sg=1,
        callbacks=[monitor]
    )  # ,negative=10)

    gensimModel.save(modelPath)
    gensimModel = gensim.models.Word2Vec.load(modelPath)
    gensimVector = gensimModel.wv;
    reducer = umap.UMAP(n_neighbors=14, min_dist=0.25, n_components=2, metric='cosine', verbose=True)
    embedding = reducer.fit_transform(gensimVector.vectors)

    keyindex = {int(entry): i for i, entry in enumerate(gensimVector.index_to_key)}
    indexkey = [keyindex[index] for index in sorted(keyindex.keys())]
    correctOriginalEmbedding = gensimVector.vectors[indexkey, :]
    correctEmbedding = embedding[indexkey, :]

    g.vs['Position'] = correctEmbedding
    xnet.igraph2xnet(g, networkPath)
    if (interactive):
        exportFilename = None

        
    def generateVisualization(correctEmbedding, gensimModel,
                              nameAttribute,sizeAttribute,
                              colorAttribute,nNeighbors,
                              legendColumns = 2, exportFilename = None):
        global groups
        
        x = correctEmbedding[:, 0]
        y = correctEmbedding[:, 1]

        names = g.vs[nameAttribute]#np.array([str(i) for i in range(len(x))])
        colorValues = g.vs[colorAttribute]
        from collections import Counter
        valuesCounter = Counter(colorValues)
        if("None" in valuesCounter):
            valuesCounter["None"] = 1
        sortedTitles = [pair[0] for pair in sorted(valuesCounter.items(), key=lambda item: item[1],reverse=True)]
        c2i = {c:i if i<10 else 10 for i,c in enumerate(sortedTitles)}
        c2cc = {c:cm.tab10(i)[0:3]+(0.25,) if i<10 else (0.75,0.75,0.75,0.05) for i,c in enumerate(sortedTitles)} 
        c2ccLight = {c:adjust_lightness(cm.tab10(i)[0:3]+(0.25,),0.98) if i<10 else adjust_lightness((0.75,0.75,0.75,1.00),0.98) for i,c in enumerate(sortedTitles)}
        colors = [c2cc[entry] for entry in colorValues];
        colorsLight = [c2ccLight[entry] for entry in colorValues];
        c = colors
        if(sizeAttribute in g.vertex_attributes()):
            sizes = 1+4*np.log(1.0+np.array(g.vs[sizeAttribute]))
        else:
            sizes = 5

        fig,ax = plt.subplots(figsize=(6,8))
        sc = plt.scatter(x,y,c=c, s=sizes, pickradius=1)

        annot = ax.annotate("", xy=(0,0), xytext=(5,5),textcoords="figure pixels",
                            bbox=dict(boxstyle="round", fc="w",ec="w"),
                            arrowprops=dict(arrowstyle="fancy",lw=0.5))
        annot.set_visible(False)
        xlim = ax.get_xlim()
        linesBack = []
        for _ in range(nNeighbors):
            line = ax.plot([0,0],[0,0],lw=4,c="k",
                           solid_capstyle='round',
                           picker=False)[0]
            line.set_visible(False)
            linesBack.append(line)

        lines = []
        for _ in range(nNeighbors):
            line = ax.plot([0,0],[0,0],lw=2,c="r",
                           solid_capstyle='round',
                           picker=False)[0]
            line.set_visible(False)
            lines.append(line)

        def update_annot(ind):
            selectedIndex = ind["ind"][0]
            annot.xy = sc.get_offsets()[selectedIndex]
            for line in lines:
                line.set_visible(False)
            for line in linesBack:
                line.set_visible(False)
            neighborsData = [(int(i),float(p)) for i,p in gensimModel.wv.most_similar(str(selectedIndex), topn=nNeighbors)]
            for i,(index,p) in enumerate(neighborsData):
                pos = (x[index],y[index])
                line = lines[i]
                lineBack = linesBack[i]
                lineData = line.get_data()
                line.set_xdata([x[selectedIndex],x[index]])
                line.set_ydata([y[selectedIndex],y[index]])
                lineBack.set_xdata([x[selectedIndex],x[index]])
                lineBack.set_ydata([y[selectedIndex],y[index]])
                line.set_visible(True)
                lineBack.set_visible(True)
                line.set_alpha(p)
                lineBack.set_alpha(p)
        #     annot.set_position((-0,-0)) + str(number) + "}$"
            text = "{}".format("\n".join([names[selectedIndex]]+[names[n] for n in [v for v,p in neighborsData]]))
            text = text.replace("$",'\$')
            annot.set_text(text)
            annot.arrow_patch.set_facecolor(c[ind["ind"][0]])
            annot.arrow_patch.set_edgecolor("w")
            annot.get_bbox_patch().set_facecolor(colorsLight[ind["ind"][0]])
            annot.get_bbox_patch().set_alpha(1.0)


        def hover(event):
            vis = annot.get_visible()
            if event.inaxes == ax:
                cont, ind = sc.contains(event)
                if cont:
                    update_annot(ind)
                    annot.set_visible(True)
                    fig.canvas.draw_idle()
                else:
                    if vis:
                        annot.set_visible(False)
                        for line in lines:
                            line.set_visible(False)
                        for line in linesBack:
                            line.set_visible(False)
                        fig.canvas.draw_idle()

        if(exportFilename is None):
            fig.canvas.mpl_connect("button_press_event", hover)

        plt.setp(ax, xticks=[], yticks=[]);
        fig.patch.set_visible(False)
        ax.axis('off')
        legend_elements = [Line2D([0], [0],
                            linewidth=0,
                            marker='o',
                            color=c2cc[community][0:3],
                            label=community,
                            # markerfacecolor='g',
                            markersize=3) for community in sortedTitles[0:10]]
        legend_elements+=[Line2D([0], [0],
                            linewidth=0,
                            marker='o',
                            color=(0.75,0.75,0.75),
                            label="Others",
                            # markerfacecolor='g',
                            markersize=3)]
        ax.legend(handles=legend_elements,
                  fontsize="small",
                  frameon=False,
                  fancybox=False,
                  ncol=legendColumns,
                  bbox_to_anchor=(0.0, 1.1),
                  loc='upper left')


        fig.subplots_adjust(bottom=0.25,top=0.90,left=0.0,right=1.0)
        if(exportFilename is None):
            plt.show()
        else:
            plt.savefig(exportFilename)
            plt.close()



    generateVisualization(
        correctEmbedding,
        gensimModel,
        nameAttribute="Paper_originalTitle",
        sizeAttribute="Paper_citationCount",
        #     colorAttribute = "JournalFixed_displayName",
        colorAttribute="Community",
        nNeighbors=10,
        legendColumns=4,
        exportFilename=exportFilename
    )




    labels = np.array(g.vs[nameAttribute])
    if(groups is None):
        lowerBound = np.percentile(sorted(g.vs["year"]),5)
        higherBound = np.percentile(sorted(g.vs["year"]),95)
        groups = {
            "Recent (>%d)"%higherBound: lambda v : v>=higherBound,
            "Past (<%d)"%lowerBound: lambda v : v<=lowerBound,
        }
    def groupMap(index):
        groupValue = g.vs[groupAttribute][index]
        for groupName,func  in groups.items():
            if(func(groupValue)):
                return groupName;
        else:
            return None # no group
    
    groupIDs = np.array(list(map(groupMap, range(g.vcount()))))
    validGroups = np.array([entry is not None for entry in groupIDs])

    if (maxSamples >= 0):
        for groupName in groups:
            groupIndices = np.where(groupIDs == groupName)[0]
            if (len(groupIndices) > maxSamples):
                validGroups[groupIndices] = False;
                validGroups[np.random.choice(groupIndices, maxSamples, replace=False)] = True

    model = emlens.SemAxis()
    model.fit(correctOriginalEmbedding[validGroups, :], groupIDs[validGroups])
    projectedCoordinates = model.transform(correctOriginalEmbedding)
    fig, ax = plt.subplots(figsize=(13, 5))
    otherGroup = np.ones(len(groupIDs), dtype=bool)
    bins = 30

    for groupName in groups:
        otherGroup *= (groupIDs != groupName)
        groupCoordinates = projectedCoordinates[groupIDs == groupName]
        if (len(groupCoordinates)):
            p = plt.hist(groupCoordinates, bins=bins, density=True, label=groupName, alpha=0.70)

    groupCoordinates = projectedCoordinates[otherGroup]
    if (len(groupCoordinates)):
        p = plt.hist(groupCoordinates, bins=bins, density=True, label="Other", alpha=0.70)

    # average

    fig.subplots_adjust(bottom=0.10, top=0.20, left=0.0, right=0.45)
    ax.legend(fontsize="small",
              frameon=False,
              fancybox=False,
              bbox_to_anchor=(0.0, 8),
              loc='upper left')

    plt.setp(ax, yticks=[]);
    fig.patch.set_visible(False)
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.set_xlabel("SemAxis")

    breaks = 15
    step = (np.max(projectedCoordinates) - np.min(projectedCoordinates)) / breaks;
    addedIndices = []
    sortedIndices = sorted(range(len(projectedCoordinates)), key=lambda i: projectedCoordinates[i])
    for i in sortedIndices:
        if (projectedCoordinates[i] >= np.min(projectedCoordinates) + len(addedIndices) * step):
            addedIndices.append(i)

    for index in addedIndices:
        #     index = 6069
        groupName = groupIDs[index]
        plt.scatter([projectedCoordinates[index]], [1.0], s=10, c="k",
                    clip_on=False,
                    transform=ax.get_xaxis_transform())
        textActor = ax.text(projectedCoordinates[index], 1.1, labels[index], fontsize=8,
                            rotation=45, rotation_mode='anchor',
                            #               transform_rotates_text=True,
                            transform=ax.get_xaxis_transform())
        textActor.set_path_effects([pe.Stroke(linewidth=2, foreground='white'),
                                    pe.Normal()])

    ax.scatter(0.2, projectedCoordinates[index])

    if (interactive):
        plt.show()
    else:
        plt.savefig(export_file_semaxis)
        plt.close()

    # ### Example analysis
    # Here we check if the semiaxis correlate with publication year

    # In[102]:

    plt.figure()
    x = g.vs["year"]
    y = projectedCoordinates
    plt.title("Pearson Corr.: %.2f, Spearman Corr.: %.2f" % (scistats.pearsonr(x, y)[0], scistats.spearmanr(x, y)[0]))
    plt.hexbin(x, y, gridsize=25, cmap=cm.inferno)
    plt.xlabel("Year")
    plt.ylabel("SemAxis")
    plt.show()
    if (interactive):
        plt.show()
    else:
        plt.savefig(export_file_correlation)
        plt.close()




def adjust_lightness(color, amount=0.5):
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], amount, c[2])
 


# ### Interactive visualization
# #### Visualization function

# In[21]:



# ### Visualization Parameters
# Choose the names of the attributes to be used for  visualization and number of neigbohrs to show.

# In[96]:



####


# ### SemAxis approach
# In this example we use a few samples of extremes for two groups (for instance past for papers published before 2000 and recent for papers published after 2015). Other classes can be used as well, for instnace journals or citations.
# Importing code directly from the emlens package (https://github.com/skojaku/emlens).

# In[15]:




if __name__ == "__main__":
    """
    How to run script:
        copy example files to input directory
        run with python line_count.py example1.csv,example2.csv /efs/input /efs/output

    What it does:
        gets the list of filenames from the commandline
        counts the lines from each file in /efs/input (which will be available within docker)
    """

    #Required cadre boilerplate to get commandline arguments:
    try:
        _input_filenames = sys.argv[1].split(',')
        _input_dir = sys.argv[2]
        _output_dir = sys.argv[3]
    except IndexError:
        print("Missing Parameter")
        sys.exit(1)
    except:
        print("Unknown Error")
        sys.exit(1)

    for input_file in _input_filenames:
        if 'edges' in input_file:
            edges_file = _input_dir + "/" + input_file
        else:
            nodes_file = _input_dir + "/" + input_file

    output_file = _output_dir + '/query_to_xnet_out.xnet'
    sentences_file = _output_dir + '/sentences.txt'
    model_file = _output_dir + '/query.model'
    export_file = _output_dir + '/embedding.pdf'
    export_file_semaxis = _output_dir + '/semaxis.pdf'
    export_file_correlation = _output_dir + '/correlation.pdf'

    mag_query_input_to_xnet(
        nodes_file,
        edges_file,
        output_file
    )
    visualize_figures(output_file, sentences_file, model_file,export_file,export_file_semaxis,export_file_correlation)



