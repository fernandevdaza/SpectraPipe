---
name: User Story
about: Historia de usuario (MVP)
title: "US-XX: <título>"
labels: ["type:story"]
---

## Story
Como <actor>, quiero <capacidad> para <beneficio>.

## Alcance
**Incluye:**
- ...
- ...

**No incluye:**
- ...
- ...

## Criterios de aceptación (AC)
- [ ] ...
- [ ] ...
- [ ] ...

## Artefactos esperados
- [ ] ...
- [ ] ...

## Errores / Edge cases
- [ ] Archivo no existe / path inválido
- [ ] Formato no soportado
- [ ] Dimensiones incompatibles
- [ ] No se puede escribir en outdir
- [ ] ...

## Demo (manual)
**Comando:**
```bash
<cli> <command> ...
```
**Resultado esperado:**

- ...
- ...

## Trazabilidad

- RF: RF-xx, RF-yy
- RNF: RNF-xx, RNF-yy
- RI: RI-xx
- CU: CU-xx

## Test plan
**Unit:**
- <módulo>: <qué valida>

**Smoke/E2E:**
- Comando: `...`
- Checks: <archivos/exit code/claves JSON>

**Manual (si aplica):**
- <pasos cortos>

## Definition of Ready (DoR)

- [ ]  AC definidos y verificables
- [ ]  Inputs/outputs claros (paths, nombres, formatos)
- [ ]  Estrategia de prueba mínima definida (smoke/unit)
- [ ]  Dependencias identificadas (si aplica)

## Tasks (JIT)

- [ ]  Implementación
- [ ]  Tests (unit/smoke si aplica)
- [ ]  Docs/README (si aplica)

## Definition of Done (DoD)

- [ ]  AC cumplidos
- [ ]  Artefactos generados y validados
- [ ]  Logs adecuados + errores accionables
- [ ]  (Si aplica) existe smoke test para este flujo y pasa
- [ ]  PR mergeado a main con checks verdes



