import typer

from .app import app

@app.command()
def bvae(ctx: typer.Context):
    """Train a model on the specified dataset."""
    typer.echo(ctx.obj)
