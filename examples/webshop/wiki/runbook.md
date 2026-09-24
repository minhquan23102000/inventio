# Nightly backup runbook

## Nightly backup

The job `nightly_backup` copies the orders database to off-site storage. It starts at 01:00
and must finish before the morning order peak at 08:00.

## When the nightly backup has not finished

1. Check the scheduler at 07:00. If `nightly_backup` is still running or has failed, stop it.
2. Take a fresh backup from the replica, not the primary: `make backup SOURCE=replica`.
3. Tell the support desk that order exports may be slow until it finishes.
4. The fresh backup keeps the date of the night it replaces, so retention counts from that night.
