# Data retention policy

## Backup retention

Nightly backups are kept for `BACKUP_RETENTION_DAYS`, then deleted. A backup is only trusted
once it has been restored: every 30 days one backup is restored to a scratch server and its
order count checked against production. While a restore test is overdue, no deploy that
changes the database schema may go out.

## Who may delete a backup

Only an engineer who did not take a backup may delete it before its retention ends. Deleting
the last backup older than a week needs a second engineer's approval.
