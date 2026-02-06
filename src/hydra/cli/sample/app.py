import typer

app = typer.Typer(help="MyCLI: a tiny example Typer app.")

@app.command()
def bvae(ctx: typer.Context):
    """Train a model on the specified dataset."""
    typer.echo(ctx.obj)

@app.command()
def ddm(ctx: typer.Context):
    """Train a conditional model on the specified dataset."""
    typer.echo(ctx.obj)
