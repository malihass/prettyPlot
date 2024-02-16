import imageio
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable


def pretty_labels(
    xlabel,
    ylabel,
    fontsize=14,
    title=None,
    grid=True,
    ax=None,
    fontname="serif",
    xminor=False,
    yminor=False,
    tight=True,
    zlabel=None,
    zminor=False,
    xticks=None,
    yticks=None,
):
    if ax is None:
        ax = plt.gca()

    ax.set_xlabel(
        xlabel,
        fontsize=fontsize,
        fontweight="bold",
        fontname=fontname,
    )
    ax.set_ylabel(
        ylabel,
        fontsize=fontsize,
        fontweight="bold",
        fontname=fontname,
    )
    if zlabel is not None:
        ax.set_zlabel(
            zlabel,
            fontsize=fontsize,
            fontweight="bold",
            fontname=fontname,
        )
    if not title is None:
        ax.set_title(
            title,
            fontsize=fontsize,
            fontweight="bold",
            fontname=fontname,
        )

    if xticks is not None:
        ax.set_xticks(xticks)
    if yticks is not None:
        ax.set_yticks(yticks)

    if zlabel is not None:
        ax.zaxis.get_offset_text().set_fontsize(fontsize)
        ax.zaxis.get_offset_text().set_fontweight("bold")
        ax.zaxis.get_offset_text().set_fontname(fontname)

    ax.yaxis.get_offset_text().set_fontsize(fontsize)
    ax.yaxis.get_offset_text().set_fontweight("bold")
    ax.yaxis.get_offset_text().set_fontname(fontname)

    ax.xaxis.get_offset_text().set_fontsize(fontsize)
    ax.xaxis.get_offset_text().set_fontweight("bold")
    ax.xaxis.get_offset_text().set_fontname(fontname)

    for tick in ax.xaxis.get_major_ticks():
        tick.label1.set_fontsize(fontsize)
        tick.label1.set_fontname(fontname)
        tick.label1.set_fontweight("bold")

    if xminor:
        for tick in ax.xaxis.get_minor_ticks():
            tick.label1.set_fontsize(fontsize)
            tick.label1.set_fontname(fontname)
            tick.label1.set_fontweight("bold")

    for tick in ax.yaxis.get_major_ticks():
        tick.label1.set_fontsize(fontsize)
        tick.label1.set_fontname(fontname)
        tick.label1.set_fontweight("bold")

    if yminor:
        for tick in ax.yaxis.get_minor_ticks():
            tick.label1.set_fontsize(fontsize)
            tick.label1.set_fontname(fontname)
            tick.label1.set_fontweight("bold")
    if zlabel is not None:
        for tick in ax.zaxis.get_major_ticks():
            tick.label1.set_fontsize(fontsize)
            tick.label1.set_fontname(fontname)
            tick.label1.set_fontweight("bold")

        if zminor:
            for tick in ax.zaxis.get_minor_ticks():
                tick.label1.set_fontsize(fontsize)
                tick.label1.set_fontname(fontname)
                tick.label1.set_fontweight("bold")

    for axis in ["top", "bottom", "left", "right"]:
        ax.spines[axis].set_linewidth(2)
        ax.spines[axis].set_color("black")

    if grid:
        ax.grid(color="k", linestyle="-", linewidth=0.5)

    if tight:
        try:
            plt.tight_layout()
        except Exception as ex:
            print(
                f"WARNING: Could not call tight_layout because: \n\n {ex} \n"
            )
            pass


def pretty_cbar(
    im=None,
    cax=None,
    label="",
    fontsize=14,
    fontsize_label=14,
    fontsize_ticks=12,
    fontname="serif",
    cbarticks=None,
):
    if fontsize is not None:
        fontsize_label = fontsize
        fontsize_ticks = fontsize - 2

    cbar = plt.colorbar(im, cax=cax, ticks=cbarticks)
    cbar.set_label(label)
    cbar_ax = cbar.ax
    text = cbar_ax.yaxis.label
    font = matplotlib.font_manager.FontProperties(
        family=fontname, weight="bold", size=fontsize_label
    )
    text.set_font_properties(font)
    for l in cbar_ax.yaxis.get_ticklabels():
        l.set_weight("bold")
        l.set_family(fontname)
        l.set_fontsize(fontsize_ticks)
    return cbar


def pretty_legend(
    ax=None,
    fontsize=13,
    fontname="serif",
    framewidth=2.0,
    **kwargs,
):
    if ax is None:
        ax = plt.gca()

    if "frameon" not in kwargs:
        kwargs["frameon"] = True
    if "loc" not in kwargs:
        kwargs["loc"] = "best"
    if "handletextpad" not in kwargs:
        kwargs["handletextpad"] = None
    if "borderpad" not in kwargs:
        kwargs["borderpad"] = None

    leg = ax.legend(
        prop={
            "family": fontname,
            "size": fontsize,
            "weight": "bold",
        },
        **kwargs,
    )
    if kwargs["frameon"]:
        leg.get_frame().set_linewidth(framewidth)
        leg.get_frame().set_edgecolor("k")


def pretty_suplabels(
    xlabel=None,
    ylabel=None,
    title=None,
    adjust_top=None,
    adjust_left=None,
    adjust_right=None,
    adjust_bottom=None,
    fontsize=14,
    fontname="serif",
    x_x=0.5,
    x_y=0.01,
    y_x=0.02,
    y_y=0.5,
    t_x=0.5,
    t_y=0.98,
):
    """Make pretty sup labels for plots

    Parameters
    ----------
    xlabel : str
        label for x abscissa
    ylabel : str
        label for y abscissa
    fontsize : int, optional
        size of the plot font, by default 14
    title : str, optional
        plot title, by default None
    adjust_top : float, optional
        where starts the top of the plot
        Useful to handle label overlap
    adjust_left : float, optional
        where starts the left of the plot
        Useful to handle label overlap
    adjust_bottom : float, optional
        where starts the bottom of plot
        Useful to handle label overlap
    """
    ax = plt.gcf()
    if xlabel is not None:
        ax.supxlabel(
            xlabel,
            fontsize=fontsize,
            fontweight="bold",
            fontname=fontname,
            x=x_x,
            y=x_y,
        )
    if ylabel is not None:
        ax.supylabel(
            ylabel,
            fontsize=fontsize,
            fontweight="bold",
            fontname=fontname,
            x=y_x,
            y=y_y,
        )
    if title is not None:
        ax.suptitle(
            title,
            fontsize=fontsize,
            fontweight="bold",
            fontname=fontname,
            x=t_x,
            y=t_y,
        )

    if (
        adjust_top is not None
        or adjust_left is not None
        or adjust_right is not None
        or adjust_bottom is not None
    ):
        ax.subplots_adjust(
            top=adjust_top,
            left=adjust_left,
            right=adjust_right,
            bottom=adjust_bottom,
        )


def pretty_bar_plot(
    xlabel1,
    yval,
    xlabel2=None,
    yerr=None,
    ymed=None,
    yerr_lower=None,
    yerr_upper=None,
    title=None,
    ylabel=None,
    bar_color=None,
    width=0.4,
    ylim=None,
    fontsize=14,
    figsize=None,
    loc="best",
    grid=True,
    fontname="serif",
    xminor=False,
    yminor=False,
):
    if ylim is not None:
        assert len(ylim) == 2

    if xlabel2 is None:
        assert len(xlabel1) == len(yval)
        if yerr is not None:
            assert len(xlabel1) == len(yerr)

        if figsize is None:
            figsize = (len(xlabel1) * 2, 6)

        fig = plt.figure(figsize=figsize)
        x = range(len(xlabel1))

        if bar_color is None:
            plt.bar(x, yval, width=width, align="center")
        else:
            plt.bar(x, yval, width=width, align="center", color=bar_color)
        if yerr is not None:
            plt.errorbar(
                x,
                yval,
                yerr,
                barsabove=True,
                capsize=5,
                elinewidth=3,
                fmt="none",
                color="k",
            )
        if (
            yerr_lower is not None
            and yerr_upper is not None
            and ymed is not None
        ):
            plt.errorbar(
                x,
                ymed,
                np.array(list(zip(yerr_lower, yerr_upper))).T,
                barsabove=True,
                capsize=5,
                elinewidth=3,
                fmt="none",
                color="k",
            )
        if ylabel is None:
            ylabel = ""
        if title is None:
            title = ""
        pretty_labels(
            "",
            ylabel,
            title=title,
            fontsize=fontsize,
            grid=grid,
            fontname=fontname,
            xminor=xminor,
            yminor=yminor,
        )
        ax = plt.gca()
        ax.set_xticks(x, xlabel1)

        for tick in ax.xaxis.get_major_ticks():
            tick.label1.set_fontsize(fontsize)
            tick.label1.set_fontname(fontname)
            tick.label1.set_fontweight("bold")
        for tick in ax.yaxis.get_major_ticks():
            tick.label1.set_fontsize(fontsize)
            tick.label1.set_fontname(fontname)
            tick.label1.set_fontweight("bold")
        for axis in ["top", "bottom", "left", "right"]:
            ax.spines[axis].set_linewidth(2)
            ax.spines[axis].set_color("black")
        if ylim is not None:
            ax.set_ylim(ylim[0], ylim[1])

    else:
        # check yval
        assert len(yval) == len(xlabel2)
        assert len(yval[xlabel2[0]]) == len(xlabel1)

        if yerr is not None:
            assert len(yerr) == len(xlabel2)
            assert len(yerr[xlabel2[0]]) == len(xlabel1)
        elif (
            yerr_lower is not None
            and yerr_upper is not None
            and ymed is not None
        ):
            assert len(yerr_lower) == len(xlabel2)
            assert len(yerr_upper) == len(xlabel2)
            assert len(ymed) == len(xlabel2)
            assert len(yerr_lower[xlabel2[0]]) == len(xlabel1)
            assert len(yerr_upper[xlabel2[0]]) == len(xlabel1)
            assert len(ymed[xlabel2[0]]) == len(xlabel1)

        x = np.arange(len(xlabel1))  # the label locations
        width = width / len(xlabel2)  # the width of the bars
        multiplier = 0

        if figsize is None:
            figsize = (len(xlabel1) * 2, 6)
        fig, ax = plt.subplots(figsize=figsize)

        if yerr is not None:
            for (lab2, measurement), (lab2, measurement_err) in zip(
                yval.items(), yerr.items()
            ):
                offset = width * multiplier
                if bar_color is None:
                    rects = ax.bar(x + offset, measurement, width, label=lab2)
                else:
                    rects = ax.bar(
                        x + offset,
                        measurement,
                        width,
                        label=lab2,
                        color=bar_color,
                    )
                ax.errorbar(
                    x + offset,
                    measurement,
                    yerr=measurement_err,
                    barsabove=True,
                    capsize=5,
                    elinewidth=3,
                    fmt="none",
                    color="k",
                )
                multiplier += 1

        elif (
            yerr_lower is not None
            and yerr_upper is not None
            and ymed is not None
        ):
            for (
                (lab2, measurement),
                (lab2, measurement_err_lo),
                (lab2, measurement_err_hi),
                (lab2, measurement_med),
            ) in zip(
                yval.items(),
                yerr_lower.items(),
                yerr_upper.items(),
                ymed.items(),
            ):
                offset = width * multiplier
                if bar_color is None:
                    rects = ax.bar(x + offset, measurement, width, label=lab2)
                else:
                    rects = ax.bar(
                        x + offset,
                        measurement,
                        width,
                        label=lab2,
                        color=bar_color[xlabel2.index(lab2)],
                    )
                multiplier += 1
                ax.errorbar(
                    x + offset,
                    measurement_med,
                    np.array(
                        list(zip(measurement_err_lo, measurement_err_hi))
                    ).T,
                    barsabove=True,
                    capsize=5,
                    elinewidth=3,
                    fmt="none",
                    color="k",
                )

        else:
            for lab2, measurement in yval.items():
                offset = width * multiplier
                if bar_color is None:
                    rects = ax.bar(x + offset, measurement, width, label=lab2)
                else:
                    rects = ax.bar(
                        x + offset,
                        measurement,
                        width,
                        label=lab2,
                        color=bar_color,
                    )
                multiplier += 1

        # Add some text for labels, title and custom x-axis tick labels, etc.
        if ylabel is None:
            ylabel = ""
        if title is None:
            title = ""
        pretty_labels(
            "",
            ylabel,
            title=title,
            fontsize=fontsize,
            grid=grid,
            fontname=fontname,
            xminor=xminor,
            yminor=yminor,
        )
        ax.set_xticks(x + width, xlabel1)

        for tick in ax.xaxis.get_major_ticks():
            tick.label1.set_fontsize(fontsize)
            tick.label1.set_fontname(fontname)
            tick.label1.set_fontweight("bold")
        for tick in ax.yaxis.get_major_ticks():
            tick.label1.set_fontsize(fontsize)
            tick.label1.set_fontname(fontname)
            tick.label1.set_fontweight("bold")
        for axis in ["top", "bottom", "left", "right"]:
            ax.spines[axis].set_linewidth(2)
            ax.spines[axis].set_color("black")

        if len(xlabel2) > 1:
            pretty_legend(fontsize=fontsize, fontname=fontname)
        if ylim is not None:
            ax.set_ylim(ylim[0], ylim[1])


def pretty_multi_contour(
    listDatax,
    listData,
    ybound,
    xbound=None,
    listTitle=None,
    listCBLabel=None,
    listXAxisName=None,
    listYAxisName=None,
    vminList=None,
    vmaxList=None,
    globalTitle=None,
    fontsize=12,
    interp="bicubic",
    xticks=None,
    yticks=None,
    figsize=None,
    grid=False,
    fontname="serif",
    log_scale_list=None,
    display_cbar_list=None,
    cbar_pad=0.2,
    cbar_size_percent=10,
    tight=True,
):
    lim = -1
    lim_vmax_t = -1
    lim_vmax_x = -1
    lim_plot = -1
    if figsize is None:
        figsize = (len(listData) * 3, 4)
    fig, axs = plt.subplots(1, len(listData), figsize=figsize)

    for i_dat in range(len(listData)):
        data = listData[i_dat]
        data_x = np.squeeze(listDatax[i_dat])
        if vminList is None:
            vmin = np.nanmin(data[:lim, :])
        else:
            vmin = vminList[i_dat]
        if vmaxList is None:
            vmax = np.nanmax(data[:lim, :])
        else:
            vmax = vmaxList[i_dat]
        if xbound is None:
            xbound = [0, data_x[-1]]
        if isinstance(ybound, float):
            ybound = [0, ybound]
        elif isinstance(ybound, int):
            ybound = [0, float(ybound)]

        if len(listData) == 1:
            loc_ax = axs
        else:
            loc_ax = axs[i_dat]

        if isinstance(listXAxisName, list):
            x_lab = listXAxisName[i_dat]
        elif isinstance(listXAxisName, str):
            x_lab = listXAxisName
        else:
            x_lab = "x"

        if isinstance(listYAxisName, list):
            y_lab = listYAxisName[i_dat]
        elif isinstance(listYAxisName, str):
            y_lab = listYAxisName
        else:
            y_lab = "t [s]"

        if isinstance(listTitle, list):
            title = listTitle[i_dat]
        elif isinstance(listTitle, str):
            title = listTitle
        else:
            title = None

        if isinstance(listCBLabel, list):
            cb_lab = listCBLabel[i_dat]
        elif isinstance(listCBLabel, str):
            cb_lab = listCBLabel
        else:
            cb_lab = ""

        if log_scale_list is None:
            log_scale = False
        else:
            log_scale = log_scale_list[i_dat]
        if vmin <= 0:
            log_scale = False

        if display_cbar_list is None:
            display_cbar = True
        else:
            display_cbar = display_cbar_list[i_dat]

        if log_scale:
            im = loc_ax.matshow(
                data[:lim, :],
                cmap=cm.viridis,
                interpolation=interp,
                extent=[xbound[0], xbound[1], ybound[1], ybound[0]],
                aspect="auto",
                norm=LogNorm(vmin, vmax),
            )
        else:
            im = loc_ax.imshow(
                data[:lim, :],
                cmap=cm.viridis,
                interpolation=interp,
                vmin=vmin,
                vmax=vmax,
                extent=[xbound[0], xbound[1], ybound[1], ybound[0]],
                aspect="auto",
            )
        divider = make_axes_locatable(loc_ax)
        if display_cbar:
            cax = divider.append_axes(
                "right", size=f"{cbar_size_percent}%", pad=cbar_pad
            )
            try:
                cbar = pretty_cbar(
                    im=im,
                    cax=cax,
                    label=cb_lab,
                    fontsize_label=fontsize,
                    fontsize_ticks=fontsize - 2,
                    fontname="serif",
                )
            except:
                print(cb_lab)
        # fig.colorbar(im, cax=cax)
        # cbar.set_label(cb_lab)
        # ax = cbar.ax
        # text = ax.yaxis.label
        # font = matplotlib.font_manager.FontProperties(
        #    family=fontname, weight="bold", size=fontsize
        # )
        # text.set_font_properties(font)

        if i_dat == 0:
            pretty_labels(
                x_lab,
                y_lab,
                fontsize,
                title=title,
                ax=loc_ax,
                grid=grid,
                fontname=fontname,
                tight=tight,
            )
        else:
            pretty_labels(
                x_lab,
                "",
                fontsize,
                title=title,
                ax=loc_ax,
                grid=grid,
                fontname=fontname,
                tight=tight,
            )

        if xticks is not None:
            loc_ax.set_xticks(xticks)
        else:
            loc_ax.set_xticks([])  # values
            loc_ax.set_xticklabels([])  # labels
        if yticks is not None and not i_dat == 0:
            loc_ax.set_yticks(yticks)
        else:
            loc_ax.set_yticks([])  # values
            loc_ax.set_yticklabels([])  # labels

        # for l in cbar.ax.yaxis.get_ticklabels():
        #    l.set_weight("bold")
        #    l.set_family(fontname)
        #    l.set_fontsize(fontsize)

    if not globalTitle is None:
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        pretty_suplabels(
            xlabel=None,
            ylabel=None,
            title=globalTitle,
            fontsize=fontsize,
            fontname=fontname,
        )


def snapVizZslice(field, x, y, figureDir, figureName, title=None):
    fig, ax = plt.subplots(1)
    plt.imshow(
        np.transpose(field),
        cmap=cm.jet,
        interpolation="bicubic",
        vmin=np.amin(field),
        vmax=np.amax(field),
        extent=[np.amin(x), np.amax(x), np.amax(y), np.amin(y)],
    )
    pretty_labels("x [m]", "y [m]", 16, title)
    plt.colorbar()
    fig.savefig(figureDir + "/" + figureName)
    plt.close(fig)
    return 0


def movieVizZslice(field, x, y, itime, movieDir, minVal=None, maxVal=None):
    fig, ax = plt.subplots(1)
    fontsize = 16
    if minVal is None:
        minVal = np.amin(field)
    if maxVal is None:
        maxVal = np.amax(field)
    plt.imshow(
        np.transpose(field),
        cmap=cm.jet,
        interpolation="bicubic",
        vmin=minVal,
        vmax=maxVal,
        extent=[np.amin(x), np.amax(x), np.amax(y), np.amin(y)],
    )
    pretty_labels("x [m]", "y [m]", 16, "Snap Id = " + str(itime))
    plt.colorbar()
    fig.savefig(movieDir + "/im_" + str(itime) + ".png")
    plt.close(fig)
    return 0


def make_movie(ntime, movieDir, movieName, prefix="im_"):
    fig = plt.figure()
    # initiate an empty  list of "plotted" images
    myimages = []
    # loops through available png:s
    for i in range(ntime):
        ## Read in picture
        fname = movieDir + "/" + prefix + str(i) + ".png"
        myimages.append(imageio.imread(fname))
    imageio.mimwrite(movieName, myimages, optimize=False)
    return


def plot_hist(field, xLabel, folder, filename):
    fig = plt.figure()
    plt.hist(field)
    fontsize = 18
    pretty_labels(xLabel, "bin count", fontsize)
    fig.savefig(folder + "/" + filename)


def plot_contour(x, y, z, color):
    ax = plt.gca()
    X, Y = np.meshgrid(x, y)
    CS = ax.contour(
        X, Y, np.transpose(z), [0.001, 0.005, 0.01, 0.05], colors=color
    )
    h, _ = CS.legend_elements()
    return h[0]


def plotActiveSubspace(paramName, W, title=None, grid=True, fontname="serif"):
    x = []
    for i, name in enumerate(paramName):
        x.append(i)
    fig = plt.figure()
    plt.bar(
        x,
        W,
        width=0.8,
        bottom=None,
        align="center",
        data=None,
        tick_label=paramName,
    )
    fontsize = 16
    if not title is None:
        plt.title(
            title,
            fontsize=fontsize,
            fontweight="bold",
            fontname=fontname,
        )
    ax = plt.gca()
    for tick in ax.xaxis.get_major_ticks():
        tick.label1.set_fontsize(fontsize)
        tick.label1.set_fontname(fontname)
        tick.label1.set_fontweight("bold")
    for tick in ax.yaxis.get_major_ticks():
        tick.label1.set_fontsize(fontsize)
        tick.label1.set_fontname(fontname)
        tick.label1.set_fontweight("bold")
    for axis in ["top", "bottom", "left", "right"]:
        ax.spines[axis].set_linewidth(2)
        ax.spines[axis].set_color("black")
    if grid:
        plt.grid(color="k", linestyle="-", linewidth=0.5)
    try:
        plt.tight_layout()
    except:
        print("Could not call tight_layout")
        pass


def plotTrainingLogs(trainingLoss, validationLoss):
    fig = plt.figure()
    plt.plot(trainingLoss, color="k", linewidth=3, label="train")
    plt.plot(validationLoss, "--", color="k", linewidth=3, label="test")
    pretty_labels("epoch", "loss", 14, title="model loss")
    pretty_legend()


def plotScatter(
    dataX, dataY, freq, title=None, xfeat=None, yfeat=None, fontSize=14
):
    fig = plt.figure()
    if xfeat is None:
        xfeat = 0
    if yfeat is None:
        yfeat = 1

    plt.plot(dataX[0::freq], dataY[0::freq], "o", color="k", markersize=3)
    if title is None:
        pretty_labels("", "", fontSize)
    else:
        pretty_labels("", "", fontSize, title=title)


def plot_probabilityMapDouble2D(
    model, minX, maxX, minY, maxY, nx=100, ny=100, minval=None, maxval=None
):
    x = np.linspace(minX, maxX, nx)
    y = np.linspace(minY, maxY, ny)
    sample = np.float64(np.zeros((nx, ny, 2)))
    for i in range(nx):
        for j in range(ny):
            sample[i, j, 0] = x[i]
            sample[i, j, 1] = y[j]
    sample = np.reshape(sample, (nx * ny, 2))
    prob = np.exp(model.log_prob(sample))
    prob = np.reshape(prob, (nx, ny))

    if minval is None:
        minval = np.amin(prob)
    if maxval is None:
        maxval = np.amax(prob)

    fig = plt.figure()
    plt.imshow(
        np.transpose(prob),
        cmap=cm.jet,
        interpolation="bicubic",
        vmin=minval,
        vmax=maxval,
        extent=[np.amin(x), np.amax(x), np.amax(y), np.amin(y)],
        aspect="auto",
    )
    plt.gca().invert_yaxis()
    plt.colorbar()
    pretty_labels(
        "1st label", "2nd label", 20, title="Approximate Probability Map"
    )


def plot_fromLatentToData(model, nSamples, xfeat=None, yfeat=None):
    if xfeat is None:
        xfeat = 0
    if yfeat is None:
        yfeat = 1
    samples = model.distribution.sample(nSamples)
    print(samples.shape)
    x, _ = model.predict(samples)
    f, axes = plt.subplots(1, 2)
    axes[0].plot(
        samples[:, xfeat], samples[:, yfeat], "o", markersize=3, color="k"
    )
    pretty_labels(
        "feature " + str(xfeat),
        "feature " + str(yfeat),
        14,
        title="Prior",
        ax=axes[0],
    )
    axes[1].plot(x[:, xfeat], x[:, yfeat], "o", markersize=3, color="k")
    pretty_labels(
        "feature " + str(xfeat),
        "feature " + str(yfeat),
        14,
        title="Generated",
        ax=axes[1],
    )
