# Isolated copy of megatron.core.utils.null_decorator (verbatim) so the NVLS
# collective code can run standalone in the benchmark harness without importing
# Megatron-LM. See docs/moe_dispatcher_deep_dive.md for provenance.


def null_decorator(*args, **kwargs):
    """
    No-op decorator.
    """
    if len(kwargs) == 0 and len(args) == 1 and callable(args[0]):
        return args[0]
    else:

        def inner(func):
            return func

        return inner
