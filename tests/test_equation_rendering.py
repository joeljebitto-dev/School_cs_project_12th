from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from app import create_equation_figure


def test_create_equation_figure_renders_mathtext_strings():
    equations = [
        r"$\Delta p = p_{\mathrm{target}} - p_{\mathrm{current}}$",
        r"$\Delta p \approx J(q)\Delta q$",
        r"$\Delta q = \alpha J^T (J J^T + \lambda^2 I)^{-1}\Delta p$",
    ]

    figure = create_equation_figure(equations, height_inches=1.25)

    assert isinstance(figure, Figure)
    assert len(figure.axes) == 1

    axis = figure.axes[0]
    assert not axis.axison
    assert len(axis.texts) == len(equations)
    assert axis.texts[2].get_text() == equations[2]

    FigureCanvasAgg(figure).draw()
