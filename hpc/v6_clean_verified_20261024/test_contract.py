"""Small contract checks for unsafe parsing and vote edge cases; no model run."""
from protocol import parse
from score import consensus

expected=['4','7']
good='[{"line_id":"4","label":"bugfix"},{"line_id":"7","label":"test"}]'
assert parse(good,expected)['predictions']=={'4':'bugfix','7':'test'}
assert not parse('```json\n'+good+'\n```',expected)['errors']
assert not parse('{"predictions":'+good+'}',expected)['errors']
assert parse('Here are labels: '+good,expected)['errors']
assert parse(good+' trailing text',expected)['errors']
assert parse(good[:-1],expected)['errors']
dup='[{"line_id":"4","label":"bugfix"},{"line_id":"4","label":"test"}]'
assert '4' not in parse(dup,expected)['predictions']
assert parse(good.replace('"test"','"made_up"'),expected)['errors']
priority=['a','b','c','d']
assert consensus(dict(a='test',b='test',c='bugfix',d='test'),priority)[:2]==('test','majority')
assert consensus(dict(a='bugfix',b='test',c='test',d='bugfix'),priority)[:2]==('bugfix','tie_priority')
assert consensus(dict(a='test',b='test',c='bugfix',d='whitespace'),priority)[:2]==('test','plurality')
assert consensus(dict(a='test',b='bugfix',c='documentation',d='whitespace'),priority)[0] is None
assert consensus(dict(a='test',b='test',c=None,d=None),priority)[0] is None
print('Parser and consensus contract checks passed')
