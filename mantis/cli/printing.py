import click
import yaml

from deprecated import deprecated


@deprecated(
    reason="This function is being moved to the biahub library, available at https://github.com/czbiohub-sf/biahub",
    action="always",
)
def echo_settings(settings):
    click.echo(yaml.dump(settings.dict(), default_flow_style=False, sort_keys=False))


@deprecated(
    reason="This function is being moved to the biahub library, available at https://github.com/czbiohub-sf/biahub",
    action="always",
)
def echo_headline(headline):
    click.echo(click.style(headline, fg="green"))
