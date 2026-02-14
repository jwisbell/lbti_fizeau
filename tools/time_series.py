import matplotlib.pyplot as plt
import pickle
import polars as pl
import polars.selectors as cs  # Great for selecting columns by type
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import numpy as np
from glob import glob
from sys import argv


def main(fdir):
    dfs = []
    ms_to_days = 1000 * 60 * 60 * 24
    # 1. Load and collect
    for file in glob(f"{fdir}/*.pkl"):
        print(f"Reading {file}...")
        with open(file, "rb") as f:
            # Load and ensure it's a Polars DataFrame
            data = pickle.load(f)
            df_temp = (
                pl.from_pandas(data) if not isinstance(data, pl.DataFrame) else data
            )

            # Add a source column just in case you need to filter later
            df_temp = df_temp.with_columns(pl.lit(file).alias("origin_file"))
            df_temp = df_temp.with_columns(
                fake_time=(
                    pl.col("PCJD")
                    .str.strip_chars()
                    .cast(pl.Float64, strict=False)
                    .min()
                    + (
                        (
                            pl.col("RAMPTIM").str.strip_chars().cast(pl.Float64)
                            / ms_to_days
                        )
                        * pl.int_range(0, pl.len())
                    )
                )
            )
            dfs.append(df_temp)

    # 2. Combine into a Master DF
    master_df = pl.concat(dfs, how="diagonal")

    # 1. Strip the strings first
    master_df = master_df.with_columns(
        cs.string().exclude("origin_file", "OBJNAME", "FILTPOLE").str.strip_chars()
    )

    # 2. Identify target columns
    # We'll create NEW float columns so you don't lose the original strings
    cols_to_fix = [
        c for c in master_df.columns if c not in ["origin_file", "OBJNAME", "FILTPOLE"]
    ]

    for col in cols_to_fix:
        master_df = master_df.with_columns(
            [
                # Attempt the cast. If it fails, it becomes null.
                pl.col(col).cast(pl.Float64, strict=False).alias(f"{col}_float")
            ]
        )

    master_df = master_df.sort("fake_time")

    master_df = master_df.with_columns(
        pl.col("central_phase_float").rolling_mean(5).alias("sma_phase")
    )

    print(master_df.head())
    for key in master_df.head():
        print(key)

    # 4. Plotting
    # target_cols = [
    #     "LBT_PARA",
    #     "LOFFSETX",
    #     "LOFFSETY",
    #     "ROFFSETX",
    #     "ROFFSETY",
    #     "WINDDIR",
    #     "WINDSPD",
    #     # "rwind1s",
    #     # "lwind1s",
    #     "LBT_AIRM",
    #     "SEEING",
    #     "central_phase",
    #     "nod_name",
    # ]  # Add your column names here
    titles = [
        "Parallactic Angle [deg]",
        "Airmass",
        'Seeing ["]',
        "Telescope Offset x [units]",
        "Telescope Offset y [units]",
        "Fourier Phase [radians]",
        "Peak Flux [counts]",
    ]

    fig = make_subplots(
        rows=len(titles),
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=titles,
    )

    # for i, col in enumerate(target_cols):
    #     if col in master_df.columns:
    col = "LBT_PARA"
    i = 0
    time = master_df["fake_time"]
    fig.add_trace(
        go.Scatter(
            # x=master_df["PCJD_float"],
            x=time,
            y=master_df[col + "_float"],
            name=col,
            mode="lines",
        ),
        row=i + 1,
        col=1,
    )
    col = "LBT_AIRM"
    i = 1
    fig.add_trace(
        go.Scatter(
            # x=master_df["PCJD_float"],
            x=time,
            y=master_df[col + "_float"],
            name=col,
            mode="lines",
        ),
        row=i + 1,
        col=1,
    )
    col = "SEEING"
    i = 2
    fig.add_trace(
        go.Scatter(
            # x=master_df["PCJD_float"],
            x=time,
            y=master_df[col + "_float"],
            name=col,
            mode="lines",
        ),
        row=i + 1,
        col=1,
    )

    col = "ROFFSETX"
    col2 = "LOFFSETX"
    i = 3
    fig.add_trace(
        go.Scatter(
            # x=master_df["PCJD_float"],
            x=time,
            y=master_df[col + "_float"],
            name=col,
            mode="lines",
            fillcolor="red",
        ),
        row=i + 1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            # x=master_df["PCJD_float"],
            x=time,
            y=master_df[col2 + "_float"],
            name=col2,
            mode="lines",
            fillcolor="blue",
        ),
        row=i + 1,
        col=1,
    )

    col = "ROFFSETY"
    col2 = "LOFFSETY"
    i = 4
    fig.add_trace(
        go.Scatter(
            # x=master_df["PCJD_float"],
            x=time,
            y=master_df[col + "_float"],
            name=col,
            mode="lines",
        ),
        row=i + 1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            # x=master_df["PCJD_float"],
            x=time,
            y=master_df[col + "_float"],
            name=col2,
            mode="lines",
        ),
        row=i + 1,
        col=1,
    )

    # maybe make this a new section... the jd isn't updated fast enough!
    col = "central_phase"
    i = 5
    fig.add_trace(
        go.Scatter(
            # x=master_df["PCJD_float"],
            x=time,
            # x=np.arange(len(master_df["PCJD_float"])),
            y=master_df[col + "_float"],
            name=col,
            mode="markers",
            # marker=dict(color="rgba(1,0,0,0.5)"),
        ),
        row=i + 1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            # x=master_df["PCJD_float"],
            x=time,
            # x=np.arange(len(master_df["PCJD_float"])),
            y=master_df["sma_phase"],
            name=col,
            mode="lines",
            line=dict(color="firebrick"),
        ),
        row=i + 1,
        col=1,
    )
    col = "flux_peak"
    i = 6
    fig.add_trace(
        go.Scatter(
            x=time,
            y=master_df[col + "_float"],
            # error_y=master_df["flux_noise_float"],
            name=col,
            mode="lines",
        ),
        row=i + 1,
        col=1,
    )

    fig.update_layout(
        height=250 * len(titles),
        template="plotly_dark",
        title_text="Observation Overview",
        showlegend=True,
    )

    fig.show()

    # fig2 = make_subplots(
    #     rows=1,
    #     cols=1,
    #     shared_xaxes=True,
    #     vertical_spacing=0.03,
    #     subplot_titles=target_cols,
    # )
    #
    # j = 3
    # rads = np.deg2rad(master_df["WINDDIR_float"].to_numpy())
    # u = master_df["WINDSPD_float"].to_numpy() * np.cos(rads)
    # v = master_df["WINDSPD_float"].to_numpy() * np.sin(rads)
    # x = master_df["PCJD_float"].to_numpy()
    # y = np.zeros(len(x))  # Keep vectors on a baseline (y=0)
    #
    # # Create quiver plot
    # quiver_fig = ff.create_quiver(
    #     x[::10],
    #     y[::10],
    #     u[::10],
    #     v[::10],
    #     scale=0.1,
    #     arrow_scale=0.05,
    #     name="Wind Vectors",
    #     line_width=2,
    # )
    # for trace in quiver_fig.data:
    #     fig2.add_trace(trace, row=1, col=1)
    # fig2.show()

    return master_df


if __name__ == "__main__":
    script, fdir = argv
    main(fdir)
