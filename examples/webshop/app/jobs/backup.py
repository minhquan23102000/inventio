BACKUP_RETENTION_DAYS = 35  # a nightly backup older than this is deleted from storage


def nightly_backup(db, storage, now):
    """Copy the orders database to off-site storage every night, before the morning order peak."""
    snapshot = db.snapshot(as_of=now)
    storage.put(f"orders-{now:%Y-%m-%d}.dump", snapshot)
    storage.delete_older_than(days=BACKUP_RETENTION_DAYS)
    return snapshot.size


def restore_test_due(last_restore_test, days=30):
    """The date by which the next restore test must have been done."""
    from datetime import timedelta

    return last_restore_test + timedelta(days=days)
