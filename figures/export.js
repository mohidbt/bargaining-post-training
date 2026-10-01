/* Render static SVG exports using the same functions as the browser. */
const fs=require('fs'),vm=require('vm'),path=require('path');
const directory=__dirname;
const sandbox={module:{exports:{}},DATA:JSON.parse(fs.readFileSync(path.join(directory,'data.json'),'utf8'))};
vm.createContext(sandbox);
vm.runInContext(fs.readFileSync(path.join(directory,'figures.js'),'utf8'),sandbox);
for(const [name,renderer] of Object.entries(sandbox.module.exports.renderers)){
 const file=path.join(directory,name+'.svg'),expected=renderer();
 if(process.argv.includes('--check')){
  if(fs.readFileSync(file,'utf8')!==expected)throw new Error(`Stale SVG export: ${name}`);
 }else fs.writeFileSync(file,expected);
}
console.log(process.argv.includes('--check')?'8 SVG exports match their current default renderers.':'Wrote 8 default SVG exports.');
