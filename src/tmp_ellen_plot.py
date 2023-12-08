
def makeDualWithinSubjectMeansPlot(
    dataFrameToPlot,
    xVariable,
    yVariable,
    fileID,
    yLims=None,
    figSize=(7, 5),
    yLabel="default_ylabel",
    xLabel="default_xlabel",
    legendTitle="default_title",
    legendLabels="default",
    myPalette=None,
    legend="auto",
):
    sns.set_context("paper", font_scale=1.9)
    #sns.set_style("whitegrid")
    sns.set_style("ticks")
    plt.figure(figsize=figSize)

    ax = sns.lineplot(
        data=dataFrameToPlot,
        x=xVariable,
        y=yVariable,
        hue="redcap_event_name",
        #errorbar=("ci", 95),
        errorbar=("se"),
        err_style="bars",
        err_kws={"capsize": 5, "elinewidth": 0.5},
        style="redcap_event_name",
        markers=["o", "D"],
        markersize=10,
        dashes=False,
        palette=myPalette,
        legend=legend,
    )

    ax.set_xticks([0, 30, 60, 90, 120, 240, 360, 420])

    if yLabel == "default_ylabel":
        ax.set_ylabel(yVariable)
    else:
        ax.set_ylabel(yLabel)

    if xLabel == "default_xlabel":
        ax.set_xlabel("")
    else:
        ax.set_xlabel(xLabel)

    if legendTitle != "default_title":
        ax.legend(title=legendTitle)

    if legendLabels != "default":
        ax.legend(labels=legendLabels)

    if yLims != None:
        ax.set_ylim(yLims[0], yLims[1])

    sns.despine(
        offset=10, trim=True
    )  # this must be above anything that messes with the ticks or labels

    filename = outputDirectory + "/PDP1_" + fileID + "_vitalswithinsubjects.png"
    plt.savefig(filename, bbox_inches="tight", dpi=600)

    filename = outputDirectory + "/PDP1_" + fileID + "_vitalswithinsubjects.pdf"
    plt.savefig(filename, transparent=False, bbox_inches="tight")
