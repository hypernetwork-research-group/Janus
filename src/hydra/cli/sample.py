import typer


@app.command()
def ddm(ctx: typer.Context):
    """Train a conditional model on the specified dataset."""
    typer.echo(ctx.obj)
