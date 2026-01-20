/* eslint-disable no-console */
export enum LogLevel {
  DEBUG = 0,
  INFO = 1,
  WARN = 2,
  ERROR = 3,
}

export class Logger {
  private static level = LogLevel.INFO;

  static setLevel(level: LogLevel): void {
    Logger.level = level;
  }

  static shouldLog(level: LogLevel): boolean {
    return level >= Logger.level;
  }

  static format(level: LogLevel, component: string, message: string): string {
    const timestamp = new Date().toISOString();
    const levelStr = LogLevel[level];
    return `${timestamp} [${levelStr}] [${component}] ${message}`;
  }

  static debug(component: string, message: string, ...args: unknown[]): void {
    if (Logger.shouldLog(LogLevel.DEBUG)) {
      const formatted = Logger.format(LogLevel.DEBUG, component, message);
      console.debug(formatted, ...args);
    }
  }

  static info(component: string, message: string, ...args: unknown[]): void {
    if (Logger.shouldLog(LogLevel.INFO)) {
      const formatted = Logger.format(LogLevel.INFO, component, message);
      console.info(formatted, ...args);
    }
  }

  static warn(component: string, message: string, ...args: unknown[]): void {
    if (Logger.shouldLog(LogLevel.WARN)) {
      const formatted = Logger.format(LogLevel.WARN, component, message);
      console.warn(formatted, ...args);
    }
  }

  static error(component: string, message: string, error?: Error | unknown): void {
    if (Logger.shouldLog(LogLevel.ERROR)) {
      const formatted = Logger.format(LogLevel.ERROR, component, message);
      if (error instanceof Error) {
        console.error(formatted, error.message, error.stack);
      } else {
        console.error(formatted, error);
      }
    }
  }
}

export class AppError extends Error {
  constructor(
    public message: string,
    public component: string,
    public code?: string,
    public cause?: Error
  ) {
    super(message);
    this.name = 'AppError';
  }
}

export function safeTry<T>(
  component: string,
  operation: string,
  fn: () => T,
  defaultValue?: T
): T {
  try {
    return fn();
  } catch (error) {
    Logger.error(component, operation, error);
    return defaultValue as T;
  }
}

export async function safeTryAsync<T>(
  component: string,
  operation: string,
  fn: () => Promise<T>,
  defaultValue?: T
): Promise<T> {
  try {
    return await fn();
  } catch (error) {
    Logger.error(component, operation, error);
    return defaultValue as T;
  }
}
