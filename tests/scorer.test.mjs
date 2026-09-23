import test from 'node:test';
import assert from 'node:assert/strict';
import {recommend, validateBundle} from '../site/scorer.mjs';
const b={schema:1,method:'mean-unit-item-cosine-v1',movies:[{id:1,title:'A'},{id:2,title:'B'},{id:3,title:'C'}],vectors:[[1,0],[0,1],[1,1]],popularity:[1,3,2]};
test('fallback, favorites, duplicates, empty eligible catalog',()=>{
  assert.deepEqual(recommend(b,[]).map(x=>x.id),[2,3,1]);
  assert.equal(recommend(b,[1])[0].id,3);
  assert.deepEqual(recommend(b,[1,1]),recommend(b,[1]));
  assert.deepEqual(recommend(b,[1,2,3]),[]);
});
test('malformed bundles and unknown IDs fail explicitly',()=>{
  assert.throws(()=>validateBundle({...b,vectors:[[NaN]]}));
  assert.throws(()=>validateBundle({...b,movies:[{id:1,title:'A'},{id:1,title:'B'},{id:3,title:'C'}]}));
  assert.throws(()=>recommend(b,[99]));
  assert.throws(()=>recommend(b,[],0));
});
