class Skip:
    """
    A special marker object used to indicate that a particular output
    from a model in a pipeline should be skipped or considered empty,
    allowing the pipeline to proceed without this specific data.
    """
    def __repr__(self):
        return "Skip()"

    def __str__(self):
        return "SkipInstance"

# Singleton instance often useful for marker objects
SKIP_INSTANCE = Skip()
