"""Build the complete current V3 report from saved evidence only."""
from .build_v3_synthesis_report import build

if __name__=='__main__':
    build(overall=True)
